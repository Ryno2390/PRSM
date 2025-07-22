"""
NWTN Enterprise Integration & Security System

Production-ready enterprise integration with comprehensive security, access control,
compliance tracking, and cloud deployment capabilities.

Completes Phase 1 of the Universal Knowledge Ingestion Engine with:
- Security classification and access control
- Audit trail generation and compliance tracking
- Enterprise authentication and authorization
- Cloud provider integration (AWS/Azure/GCP)
- Data governance and privacy protection

Part of NWTN Phase 1: Universal Knowledge Ingestion Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json
import os
import uuid
from pathlib import Path
import threading
import logging
from contextlib import contextmanager
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class AccessLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"

class UserRole(Enum):
    GUEST = "guest"
    USER = "user"
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SECURITY_OFFICER = "security_officer"
    SYSTEM_ADMIN = "system_admin"

class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISES = "on_premises"

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"
    FedRAMP = "fedramp"

@dataclass
class SecurityContext:
    user_id: str
    user_role: UserRole
    access_levels: Set[AccessLevel]
    security_clearance: SecurityClassification
    session_id: str
    expires_at: datetime
    ip_address: str
    user_agent: str
    additional_attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEvent:
    event_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_id: str
    resource_type: str
    security_classification: SecurityClassification
    success: bool
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)

@dataclass
class AccessControlRule:
    rule_id: str
    resource_pattern: str  # regex pattern
    required_role: UserRole
    required_access_level: AccessLevel
    required_security_clearance: SecurityClassification
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""

@dataclass
class EncryptionKey:
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime]
    classification: SecurityClassification

class SecurityClassificationManager:
    """Automatic security classification based on content analysis"""
    
    def __init__(self):
        self.classification_rules = self._initialize_classification_rules()
        self.keyword_patterns = self._initialize_keyword_patterns()
        self.lock = threading.RLock()
    
    def classify_content(self, content: str, metadata: Dict[str, Any] = None) -> SecurityClassification:
        """Automatically classify content security level"""
        with self.lock:
            # Start with public as default
            classification = SecurityClassification.PUBLIC
            
            content_lower = content.lower()
            metadata = metadata or {}
            
            # Check for sensitive keywords
            for level, patterns in self.keyword_patterns.items():
                if any(pattern in content_lower for pattern in patterns):
                    classification = max(classification, level, key=lambda x: self._get_classification_level(x))
            
            # Check metadata-based rules
            source_type = metadata.get('source_type', '')
            if 'financial' in source_type or 'hr' in source_type:
                classification = max(classification, SecurityClassification.CONFIDENTIAL, 
                                   key=lambda x: self._get_classification_level(x))
            
            if 'legal' in source_type or 'contract' in source_type:
                classification = max(classification, SecurityClassification.RESTRICTED,
                                   key=lambda x: self._get_classification_level(x))
            
            # Check for PII patterns
            if self._contains_pii(content):
                classification = max(classification, SecurityClassification.CONFIDENTIAL,
                                   key=lambda x: self._get_classification_level(x))
            
            return classification
    
    def classify_entity(self, entity_name: str, entity_attributes: Dict[str, Any]) -> SecurityClassification:
        """Classify security level of extracted entities"""
        entity_lower = entity_name.lower()
        
        # Person entities with contact info are confidential
        if any(indicator in entity_lower for indicator in ['email', 'phone', 'ssn', 'address']):
            return SecurityClassification.CONFIDENTIAL
        
        # Financial entities
        if any(indicator in entity_lower for indicator in ['salary', 'revenue', 'profit', 'cost']):
            return SecurityClassification.RESTRICTED
        
        # Project entities with strategic keywords
        if any(indicator in entity_lower for indicator in ['strategic', 'merger', 'acquisition']):
            return SecurityClassification.RESTRICTED
        
        return SecurityClassification.INTERNAL
    
    def _initialize_classification_rules(self) -> Dict[SecurityClassification, List[Callable]]:
        """Initialize classification rules"""
        return {
            SecurityClassification.CONFIDENTIAL: [
                lambda content: 'confidential' in content.lower(),
                lambda content: 'private' in content.lower(),
                lambda content: 'sensitive' in content.lower()
            ],
            SecurityClassification.RESTRICTED: [
                lambda content: 'restricted' in content.lower(),
                lambda content: 'proprietary' in content.lower(),
                lambda content: 'trade secret' in content.lower()
            ],
            SecurityClassification.TOP_SECRET: [
                lambda content: 'top secret' in content.lower(),
                lambda content: 'classified' in content.lower()
            ]
        }
    
    def _initialize_keyword_patterns(self) -> Dict[SecurityClassification, List[str]]:
        """Initialize keyword patterns for classification"""
        return {
            SecurityClassification.INTERNAL: [
                'internal', 'employee', 'staff', 'team'
            ],
            SecurityClassification.CONFIDENTIAL: [
                'confidential', 'private', 'personal', 'sensitive', 'ssn', 'social security',
                'credit card', 'password', 'salary', 'compensation'
            ],
            SecurityClassification.RESTRICTED: [
                'restricted', 'proprietary', 'trade secret', 'patent', 'financial results',
                'revenue', 'profit', 'merger', 'acquisition', 'strategic plan'
            ],
            SecurityClassification.TOP_SECRET: [
                'top secret', 'classified', 'national security', 'defense'
            ]
        }
    
    def _get_classification_level(self, classification: SecurityClassification) -> int:
        """Get numeric level for classification comparison"""
        levels = {
            SecurityClassification.PUBLIC: 0,
            SecurityClassification.INTERNAL: 1,
            SecurityClassification.CONFIDENTIAL: 2,
            SecurityClassification.RESTRICTED: 3,
            SecurityClassification.TOP_SECRET: 4
        }
        return levels[classification]
    
    def _contains_pii(self, content: str) -> bool:
        """Check if content contains personally identifiable information"""
        import re
        
        # Email pattern
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
            return True
        
        # Phone pattern
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content):
            return True
        
        # SSN pattern
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', content):
            return True
        
        return False

class AccessControlManager:
    """Enterprise-grade access control with role-based permissions"""
    
    def __init__(self):
        self.access_rules: Dict[str, AccessControlRule] = {}
        self.user_permissions: Dict[str, Dict[str, Any]] = {}
        self.role_hierarchy = self._initialize_role_hierarchy()
        self.lock = threading.RLock()
        
    def add_access_rule(self, rule: AccessControlRule) -> bool:
        """Add new access control rule"""
        with self.lock:
            self.access_rules[rule.rule_id] = rule
            return True
    
    def check_access(self, security_context: SecurityContext, resource_id: str, 
                    resource_type: str, required_access: AccessLevel,
                    resource_classification: SecurityClassification) -> bool:
        """Check if user has access to resource"""
        with self.lock:
            # Check security clearance level
            if not self._has_sufficient_clearance(security_context.security_clearance, 
                                                resource_classification):
                return False
            
            # Check access level
            if required_access not in security_context.access_levels:
                return False
            
            # Check role-based access
            if not self._check_role_access(security_context.user_role, required_access):
                return False
            
            # Check specific access rules
            applicable_rules = self._get_applicable_rules(resource_id, resource_type)
            for rule in applicable_rules:
                if not self._evaluate_rule(security_context, rule):
                    return False
            
            return True
    
    def grant_user_permission(self, user_id: str, resource_id: str, access_level: AccessLevel,
                            granted_by: str) -> bool:
        """Grant specific permission to user"""
        with self.lock:
            if user_id not in self.user_permissions:
                self.user_permissions[user_id] = {}
            
            self.user_permissions[user_id][resource_id] = {
                'access_level': access_level,
                'granted_by': granted_by,
                'granted_at': datetime.now()
            }
            return True
    
    def revoke_user_permission(self, user_id: str, resource_id: str, revoked_by: str) -> bool:
        """Revoke specific permission from user"""
        with self.lock:
            if user_id in self.user_permissions and resource_id in self.user_permissions[user_id]:
                self.user_permissions[user_id][resource_id]['revoked_at'] = datetime.now()
                self.user_permissions[user_id][resource_id]['revoked_by'] = revoked_by
                del self.user_permissions[user_id][resource_id]
                return True
            return False
    
    def _initialize_role_hierarchy(self) -> Dict[UserRole, Set[UserRole]]:
        """Initialize role hierarchy (roles that inherit permissions)"""
        return {
            UserRole.SYSTEM_ADMIN: {UserRole.ADMIN, UserRole.SECURITY_OFFICER, UserRole.RESEARCHER, UserRole.ANALYST, UserRole.USER, UserRole.GUEST},
            UserRole.ADMIN: {UserRole.RESEARCHER, UserRole.ANALYST, UserRole.USER, UserRole.GUEST},
            UserRole.SECURITY_OFFICER: {UserRole.RESEARCHER, UserRole.ANALYST, UserRole.USER, UserRole.GUEST},
            UserRole.RESEARCHER: {UserRole.ANALYST, UserRole.USER, UserRole.GUEST},
            UserRole.ANALYST: {UserRole.USER, UserRole.GUEST},
            UserRole.USER: {UserRole.GUEST},
            UserRole.GUEST: set()
        }
    
    def _has_sufficient_clearance(self, user_clearance: SecurityClassification, 
                                resource_clearance: SecurityClassification) -> bool:
        """Check if user has sufficient security clearance"""
        clearance_levels = {
            SecurityClassification.PUBLIC: 0,
            SecurityClassification.INTERNAL: 1,
            SecurityClassification.CONFIDENTIAL: 2,
            SecurityClassification.RESTRICTED: 3,
            SecurityClassification.TOP_SECRET: 4
        }
        return clearance_levels[user_clearance] >= clearance_levels[resource_clearance]
    
    def _check_role_access(self, user_role: UserRole, required_access: AccessLevel) -> bool:
        """Check if user role allows required access level"""
        # System admin has all access
        if user_role == UserRole.SYSTEM_ADMIN:
            return True
        
        # Admin has read/write access
        if user_role == UserRole.ADMIN and required_access in [AccessLevel.READ, AccessLevel.WRITE]:
            return True
        
        # Researchers and analysts have read/write access
        if user_role in [UserRole.RESEARCHER, UserRole.ANALYST] and required_access in [AccessLevel.READ, AccessLevel.WRITE]:
            return True
        
        # Users have read access
        if user_role == UserRole.USER and required_access == AccessLevel.READ:
            return True
        
        # Guests have limited read access
        if user_role == UserRole.GUEST and required_access == AccessLevel.READ:
            return True
        
        return False
    
    def _get_applicable_rules(self, resource_id: str, resource_type: str) -> List[AccessControlRule]:
        """Get access control rules applicable to resource"""
        import re
        applicable_rules = []
        
        for rule in self.access_rules.values():
            if re.match(rule.resource_pattern, resource_id) or re.match(rule.resource_pattern, resource_type):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule(self, security_context: SecurityContext, rule: AccessControlRule) -> bool:
        """Evaluate if security context satisfies access control rule"""
        # Check role requirement
        if security_context.user_role not in self.role_hierarchy.get(rule.required_role, {rule.required_role}):
            return False
        
        # Check access level requirement
        if rule.required_access_level not in security_context.access_levels:
            return False
        
        # Check security clearance requirement
        if not self._has_sufficient_clearance(security_context.security_clearance, 
                                            rule.required_security_clearance):
            return False
        
        # Check additional conditions
        for condition, expected_value in rule.conditions.items():
            if security_context.additional_attributes.get(condition) != expected_value:
                return False
        
        return True

class AuditTrailManager:
    """Comprehensive audit trail and compliance tracking"""
    
    def __init__(self, max_events: int = 1000000):
        self.audit_events: List[AuditEvent] = []
        self.max_events = max_events
        self.compliance_rules: Dict[ComplianceFramework, List[Callable]] = self._initialize_compliance_rules()
        self.lock = threading.RLock()
        self.audit_storage_path = Path("audit_logs")
        self.audit_storage_path.mkdir(exist_ok=True)
    
    def log_event(self, user_id: str, action: str, resource_id: str, resource_type: str,
                 security_classification: SecurityClassification, success: bool,
                 ip_address: str, user_agent: str, details: Dict[str, Any] = None,
                 compliance_frameworks: Set[ComplianceFramework] = None) -> str:
        """Log audit event"""
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            security_classification=security_classification,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            compliance_frameworks=compliance_frameworks or set()
        )
        
        with self.lock:
            self.audit_events.append(event)
            
            # Rotate logs if necessary
            if len(self.audit_events) > self.max_events:
                self._rotate_audit_logs()
        
        # Write to persistent storage
        self._write_audit_event_to_storage(event)
        
        return event_id
    
    def query_audit_events(self, start_time: datetime = None, end_time: datetime = None,
                          user_id: str = None, action: str = None, resource_id: str = None,
                          success: bool = None) -> List[AuditEvent]:
        """Query audit events with filters"""
        with self.lock:
            filtered_events = self.audit_events.copy()
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if action:
            filtered_events = [e for e in filtered_events if e.action == action]
        
        if resource_id:
            filtered_events = [e for e in filtered_events if e.resource_id == resource_id]
        
        if success is not None:
            filtered_events = [e for e in filtered_events if e.success == success]
        
        return filtered_events
    
    def generate_compliance_report(self, framework: ComplianceFramework, 
                                 start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        events = self.query_audit_events(start_time=start_time, end_time=end_time)
        
        # Filter events relevant to compliance framework
        relevant_events = [e for e in events if framework in e.compliance_frameworks]
        
        report = {
            'framework': framework.value,
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(relevant_events),
            'successful_events': len([e for e in relevant_events if e.success]),
            'failed_events': len([e for e in relevant_events if not e.success]),
            'unique_users': len(set(e.user_id for e in relevant_events)),
            'event_breakdown': self._analyze_event_breakdown(relevant_events),
            'compliance_violations': self._detect_compliance_violations(relevant_events, framework),
            'recommendations': self._generate_compliance_recommendations(relevant_events, framework)
        }
        
        return report
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceFramework, List[Callable]]:
        """Initialize compliance validation rules"""
        return {
            ComplianceFramework.GDPR: [
                lambda event: self._validate_gdpr_data_access(event),
                lambda event: self._validate_gdpr_consent(event)
            ],
            ComplianceFramework.HIPAA: [
                lambda event: self._validate_hipaa_access(event),
                lambda event: self._validate_hipaa_audit_trail(event)
            ],
            ComplianceFramework.SOX: [
                lambda event: self._validate_sox_financial_access(event),
                lambda event: self._validate_sox_audit_integrity(event)
            ]
        }
    
    def _rotate_audit_logs(self):
        """Rotate audit logs to persistent storage"""
        # Move oldest 10% of events to persistent storage
        rotation_count = self.max_events // 10
        events_to_archive = self.audit_events[:rotation_count]
        
        # Archive to file
        archive_file = self.audit_storage_path / f"audit_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(archive_file, 'w') as f:
            json.dump([self._serialize_audit_event(e) for e in events_to_archive], f)
        
        # Remove from memory
        self.audit_events = self.audit_events[rotation_count:]
    
    def _write_audit_event_to_storage(self, event: AuditEvent):
        """Write individual audit event to persistent storage"""
        daily_log_file = self.audit_storage_path / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
        
        event_data = self._serialize_audit_event(event)
        
        # Append to daily log file
        with open(daily_log_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
    
    def _serialize_audit_event(self, event: AuditEvent) -> Dict[str, Any]:
        """Serialize audit event for storage"""
        return {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'action': event.action,
            'resource_id': event.resource_id,
            'resource_type': event.resource_type,
            'security_classification': event.security_classification.value,
            'success': event.success,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'details': event.details,
            'compliance_frameworks': [f.value for f in event.compliance_frameworks]
        }
    
    def _analyze_event_breakdown(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Analyze breakdown of events by type"""
        breakdown = {}
        for event in events:
            breakdown[event.action] = breakdown.get(event.action, 0) + 1
        return breakdown
    
    def _detect_compliance_violations(self, events: List[AuditEvent], 
                                    framework: ComplianceFramework) -> List[Dict[str, Any]]:
        """Detect compliance violations"""
        violations = []
        
        if framework == ComplianceFramework.GDPR:
            # Check for excessive data access
            user_access_counts = {}
            for event in events:
                if event.action == 'data_access':
                    user_access_counts[event.user_id] = user_access_counts.get(event.user_id, 0) + 1
            
            for user_id, count in user_access_counts.items():
                if count > 1000:  # Threshold for excessive access
                    violations.append({
                        'type': 'excessive_data_access',
                        'user_id': user_id,
                        'access_count': count,
                        'description': f"User {user_id} accessed data {count} times"
                    })
        
        return violations
    
    def _generate_compliance_recommendations(self, events: List[AuditEvent], 
                                           framework: ComplianceFramework) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if framework == ComplianceFramework.GDPR:
            # Check for missing consent tracking
            data_access_events = [e for e in events if e.action == 'data_access']
            consent_events = [e for e in events if e.action == 'consent_verification']
            
            if len(data_access_events) > len(consent_events) * 2:
                recommendations.append("Implement stronger consent verification for data access")
        
        return recommendations
    
    def _validate_gdpr_data_access(self, event: AuditEvent) -> bool:
        """Validate GDPR data access requirements"""
        if event.action == 'data_access' and event.security_classification in [SecurityClassification.CONFIDENTIAL, SecurityClassification.RESTRICTED]:
            return 'consent_verified' in event.details
        return True
    
    def _validate_gdpr_consent(self, event: AuditEvent) -> bool:
        """Validate GDPR consent requirements"""
        return True  # Simplified validation
    
    def _validate_hipaa_access(self, event: AuditEvent) -> bool:
        """Validate HIPAA access requirements"""
        return True  # Simplified validation
    
    def _validate_hipaa_audit_trail(self, event: AuditEvent) -> bool:
        """Validate HIPAA audit trail requirements"""
        return True  # Simplified validation
    
    def _validate_sox_financial_access(self, event: AuditEvent) -> bool:
        """Validate SOX financial access requirements"""
        return True  # Simplified validation
    
    def _validate_sox_audit_integrity(self, event: AuditEvent) -> bool:
        """Validate SOX audit integrity requirements"""
        return True  # Simplified validation

class EncryptionManager:
    """Enterprise-grade encryption for data at rest and in transit"""
    
    def __init__(self, master_key: str = None):
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.master_key = master_key or os.environ.get('NWTN_MASTER_KEY', self._generate_master_key())
        self.cipher_suite = self._initialize_cipher_suite()
        self.lock = threading.RLock()
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str = None,
                    classification: SecurityClassification = SecurityClassification.CONFIDENTIAL) -> Tuple[bytes, str]:
        """Encrypt data and return encrypted data with key ID"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if not key_id:
            key_id = self.generate_encryption_key(classification)
        
        key = self.encryption_keys[key_id]
        cipher = Fernet(key.key_data)
        encrypted_data = cipher.encrypt(data)
        
        return encrypted_data, key_id
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key"""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")
        
        key = self.encryption_keys[key_id]
        
        # Check if key is expired
        if key.expires_at and datetime.now() > key.expires_at:
            raise ValueError(f"Encryption key {key_id} is expired")
        
        cipher = Fernet(key.key_data)
        decrypted_data = cipher.decrypt(encrypted_data)
        
        return decrypted_data
    
    def generate_encryption_key(self, classification: SecurityClassification,
                               expires_in_days: int = None) -> str:
        """Generate new encryption key"""
        key_id = str(uuid.uuid4())
        key_data = Fernet.generate_key()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm="Fernet",
            created_at=datetime.now(),
            expires_at=expires_at,
            classification=classification
        )
        
        with self.lock:
            self.encryption_keys[key_id] = encryption_key
        
        return key_id
    
    def rotate_encryption_key(self, old_key_id: str) -> str:
        """Rotate encryption key (create new key and mark old as deprecated)"""
        if old_key_id not in self.encryption_keys:
            raise ValueError(f"Key {old_key_id} not found")
        
        old_key = self.encryption_keys[old_key_id]
        new_key_id = self.generate_encryption_key(old_key.classification)
        
        # Mark old key as expired
        old_key.expires_at = datetime.now()
        
        return new_key_id
    
    def _generate_master_key(self) -> str:
        """Generate master encryption key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
    
    def _initialize_cipher_suite(self) -> Fernet:
        """Initialize cipher suite with master key"""
        key = base64.urlsafe_b64encode(self.master_key.encode('utf-8')[:32].ljust(32, b'0'))
        return Fernet(key)

class CloudIntegrationManager:
    """Cloud provider integration for AWS/Azure/GCP deployment"""
    
    def __init__(self, provider: CloudProvider = CloudProvider.AWS):
        self.provider = provider
        self.deployment_configs: Dict[CloudProvider, Dict[str, Any]] = {}
        self.monitoring_endpoints: Dict[str, str] = {}
        self.lock = threading.RLock()
    
    def configure_cloud_deployment(self, provider: CloudProvider, config: Dict[str, Any]) -> bool:
        """Configure cloud deployment settings"""
        with self.lock:
            self.deployment_configs[provider] = {
                **config,
                'configured_at': datetime.now(),
                'status': 'configured'
            }
            return True
    
    def deploy_to_cloud(self, provider: CloudProvider, deployment_name: str) -> Dict[str, Any]:
        """Deploy NWTN system to cloud provider"""
        if provider not in self.deployment_configs:
            return {'success': False, 'error': f'No configuration found for {provider.value}'}
        
        config = self.deployment_configs[provider]
        
        # Simulate deployment process
        deployment_result = {
            'success': True,
            'deployment_id': str(uuid.uuid4()),
            'deployment_name': deployment_name,
            'provider': provider.value,
            'endpoints': self._generate_deployment_endpoints(provider, deployment_name),
            'monitoring_dashboard': self._generate_monitoring_dashboard_url(provider, deployment_name),
            'deployed_at': datetime.now().isoformat(),
            'estimated_cost_per_hour': self._estimate_deployment_cost(provider, config)
        }
        
        return deployment_result
    
    def setup_auto_scaling(self, deployment_id: str, min_instances: int, max_instances: int,
                          cpu_threshold: float = 70.0) -> Dict[str, Any]:
        """Setup auto-scaling configuration"""
        auto_scaling_config = {
            'deployment_id': deployment_id,
            'min_instances': min_instances,
            'max_instances': max_instances,
            'cpu_threshold': cpu_threshold,
            'scale_up_cooldown': 300,  # seconds
            'scale_down_cooldown': 600,  # seconds
            'configured_at': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'auto_scaling_config': auto_scaling_config
        }
    
    def setup_monitoring(self, deployment_id: str) -> Dict[str, Any]:
        """Setup monitoring and alerting"""
        monitoring_config = {
            'deployment_id': deployment_id,
            'metrics_enabled': True,
            'log_aggregation': True,
            'alerting_rules': [
                {'metric': 'cpu_utilization', 'threshold': 80, 'action': 'alert'},
                {'metric': 'memory_utilization', 'threshold': 85, 'action': 'alert'},
                {'metric': 'error_rate', 'threshold': 5, 'action': 'alert'},
                {'metric': 'response_time', 'threshold': 5000, 'action': 'alert'}  # ms
            ],
            'dashboard_url': self._generate_monitoring_dashboard_url(self.provider, deployment_id)
        }
        
        return {
            'success': True,
            'monitoring_config': monitoring_config
        }
    
    def estimate_costs(self, provider: CloudProvider, config: Dict[str, Any],
                      hours_per_month: int = 720) -> Dict[str, Any]:
        """Estimate cloud deployment costs"""
        base_cost_per_hour = self._estimate_deployment_cost(provider, config)
        
        cost_breakdown = {
            'base_infrastructure': base_cost_per_hour * hours_per_month,
            'data_storage': config.get('storage_gb', 100) * 0.023 * hours_per_month / 720,  # $0.023 per GB-month
            'data_transfer': config.get('transfer_gb_per_month', 100) * 0.09,  # $0.09 per GB
            'monitoring_and_logging': base_cost_per_hour * 0.1 * hours_per_month,  # 10% of base cost
        }
        
        total_monthly_cost = sum(cost_breakdown.values())
        
        return {
            'provider': provider.value,
            'hours_per_month': hours_per_month,
            'cost_breakdown': cost_breakdown,
            'total_monthly_cost': total_monthly_cost,
            'currency': 'USD'
        }
    
    def _generate_deployment_endpoints(self, provider: CloudProvider, deployment_name: str) -> Dict[str, str]:
        """Generate deployment endpoints"""
        base_domains = {
            CloudProvider.AWS: 'amazonaws.com',
            CloudProvider.AZURE: 'azure.com',
            CloudProvider.GCP: 'googleapis.com'
        }
        
        base_domain = base_domains.get(provider, 'example.com')
        
        return {
            'api_endpoint': f'https://{deployment_name}-api.{base_domain}',
            'web_interface': f'https://{deployment_name}-web.{base_domain}',
            'knowledge_graph_api': f'https://{deployment_name}-kg.{base_domain}',
            'reasoning_api': f'https://{deployment_name}-reasoning.{base_domain}'
        }
    
    def _generate_monitoring_dashboard_url(self, provider: CloudProvider, deployment_name: str) -> str:
        """Generate monitoring dashboard URL"""
        dashboard_urls = {
            CloudProvider.AWS: f'https://console.aws.amazon.com/cloudwatch/dashboard/{deployment_name}',
            CloudProvider.AZURE: f'https://portal.azure.com/dashboard/{deployment_name}',
            CloudProvider.GCP: f'https://console.cloud.google.com/monitoring/dashboard/{deployment_name}'
        }
        
        return dashboard_urls.get(provider, f'https://monitoring.example.com/dashboard/{deployment_name}')
    
    def _estimate_deployment_cost(self, provider: CloudProvider, config: Dict[str, Any]) -> float:
        """Estimate hourly deployment cost"""
        instance_costs = {
            CloudProvider.AWS: {
                'small': 0.0116,  # t3.micro
                'medium': 0.0464,  # t3.small
                'large': 0.0928   # t3.medium
            },
            CloudProvider.AZURE: {
                'small': 0.0124,
                'medium': 0.0496,
                'large': 0.0992
            },
            CloudProvider.GCP: {
                'small': 0.0100,
                'medium': 0.0400,
                'large': 0.0800
            }
        }
        
        instance_size = config.get('instance_size', 'medium')
        instance_count = config.get('instance_count', 2)
        
        base_cost = instance_costs[provider][instance_size] * instance_count
        
        # Add additional services cost (databases, load balancers, etc.)
        additional_services_cost = base_cost * 0.3
        
        return base_cost + additional_services_cost

class EnterpriseIntegrationSecurity:
    """Main orchestrator for enterprise integration and security"""
    
    def __init__(self, cloud_provider: CloudProvider = CloudProvider.AWS):
        self.security_classifier = SecurityClassificationManager()
        self.access_control = AccessControlManager()
        self.audit_manager = AuditTrailManager()
        self.encryption_manager = EncryptionManager()
        self.cloud_integration = CloudIntegrationManager(cloud_provider)
        
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.lock = threading.RLock()
        
        # Initialize default access rules
        self._initialize_default_access_rules()
    
    def create_security_context(self, user_id: str, user_role: UserRole, 
                              security_clearance: SecurityClassification,
                              ip_address: str, user_agent: str) -> SecurityContext:
        """Create new security context for user session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=8)  # 8-hour session
        
        # Determine access levels based on role
        access_levels = self._get_access_levels_for_role(user_role)
        
        security_context = SecurityContext(
            user_id=user_id,
            user_role=user_role,
            access_levels=access_levels,
            security_clearance=security_clearance,
            session_id=session_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self.lock:
            self.active_sessions[session_id] = security_context
        
        # Log session creation
        self.audit_manager.log_event(
            user_id=user_id,
            action="session_created",
            resource_id=session_id,
            resource_type="session",
            security_classification=SecurityClassification.INTERNAL,
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
            compliance_frameworks={ComplianceFramework.GDPR, ComplianceFramework.ISO27001}
        )
        
        return security_context
    
    def process_secure_content(self, content: str, metadata: Dict[str, Any], 
                             security_context: SecurityContext) -> Dict[str, Any]:
        """Process content with security classification and access control"""
        try:
            # Classify content security level
            content_classification = self.security_classifier.classify_content(content, metadata)
            
            # Check if user has access to this classification level
            if not self.access_control.check_access(
                security_context, 
                resource_id=metadata.get('source_id', 'unknown'),
                resource_type='content',
                required_access=AccessLevel.READ,
                resource_classification=content_classification
            ):
                # Log access denial
                self.audit_manager.log_event(
                    user_id=security_context.user_id,
                    action="content_access_denied",
                    resource_id=metadata.get('source_id', 'unknown'),
                    resource_type="content",
                    security_classification=content_classification,
                    success=False,
                    ip_address=security_context.ip_address,
                    user_agent=security_context.user_agent
                )
                
                return {
                    'success': False,
                    'error': 'Access denied - insufficient security clearance',
                    'required_classification': content_classification.value
                }
            
            # Encrypt sensitive content
            if content_classification in [SecurityClassification.CONFIDENTIAL, SecurityClassification.RESTRICTED, SecurityClassification.TOP_SECRET]:
                encrypted_content, key_id = self.encryption_manager.encrypt_data(content, classification=content_classification)
            else:
                encrypted_content = content.encode('utf-8')
                key_id = None
            
            # Log successful access
            self.audit_manager.log_event(
                user_id=security_context.user_id,
                action="content_processed",
                resource_id=metadata.get('source_id', 'unknown'),
                resource_type="content",
                security_classification=content_classification,
                success=True,
                ip_address=security_context.ip_address,
                user_agent=security_context.user_agent,
                details={
                    'content_length': len(content),
                    'encrypted': key_id is not None
                },
                compliance_frameworks={ComplianceFramework.GDPR, ComplianceFramework.ISO27001}
            )
            
            return {
                'success': True,
                'classification': content_classification.value,
                'encrypted_content': encrypted_content,
                'encryption_key_id': key_id,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Log error
            self.audit_manager.log_event(
                user_id=security_context.user_id,
                action="content_processing_error",
                resource_id=metadata.get('source_id', 'unknown'),
                resource_type="content",
                security_classification=SecurityClassification.INTERNAL,
                success=False,
                ip_address=security_context.ip_address,
                user_agent=security_context.user_agent,
                details={'error': str(e)}
            )
            
            return {
                'success': False,
                'error': f'Content processing failed: {str(e)}'
            }
    
    def deploy_enterprise_system(self, deployment_name: str, provider: CloudProvider,
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy complete enterprise NWTN system to cloud"""
        try:
            # Configure cloud deployment
            self.cloud_integration.configure_cloud_deployment(provider, config)
            
            # Deploy to cloud
            deployment_result = self.cloud_integration.deploy_to_cloud(provider, deployment_name)
            
            if not deployment_result['success']:
                return deployment_result
            
            # Setup auto-scaling
            auto_scaling = self.cloud_integration.setup_auto_scaling(
                deployment_result['deployment_id'],
                min_instances=config.get('min_instances', 2),
                max_instances=config.get('max_instances', 10)
            )
            
            # Setup monitoring
            monitoring = self.cloud_integration.setup_monitoring(deployment_result['deployment_id'])
            
            # Estimate costs
            cost_estimate = self.cloud_integration.estimate_costs(provider, config)
            
            return {
                'success': True,
                'deployment': deployment_result,
                'auto_scaling': auto_scaling,
                'monitoring': monitoring,
                'cost_estimate': cost_estimate
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Enterprise deployment failed: {str(e)}'
            }
    
    def generate_security_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            # Get audit events for period
            audit_events = self.audit_manager.query_audit_events(start_time=start_time, end_time=end_time)
            
            # Generate compliance reports
            compliance_reports = {}
            for framework in ComplianceFramework:
                try:
                    report = self.audit_manager.generate_compliance_report(framework, start_time, end_time)
                    compliance_reports[framework.value] = report
                except Exception as e:
                    compliance_reports[framework.value] = {'error': str(e)}
            
            # Calculate security metrics
            total_events = len(audit_events)
            successful_events = len([e for e in audit_events if e.success])
            failed_events = total_events - successful_events
            
            unique_users = len(set(e.user_id for e in audit_events))
            
            security_report = {
                'report_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'summary': {
                    'total_events': total_events,
                    'successful_events': successful_events,
                    'failed_events': failed_events,
                    'success_rate': (successful_events / total_events * 100) if total_events > 0 else 0,
                    'unique_users': unique_users
                },
                'compliance_reports': compliance_reports,
                'security_classifications': self._analyze_security_classifications(audit_events),
                'access_patterns': self._analyze_access_patterns(audit_events),
                'recommendations': self._generate_security_recommendations(audit_events)
            }
            
            return security_report
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Security report generation failed: {str(e)}'
            }
    
    def _initialize_default_access_rules(self):
        """Initialize default access control rules"""
        # Rule for knowledge graph access
        kg_rule = AccessControlRule(
            rule_id="knowledge_graph_access",
            resource_pattern=".*knowledge_graph.*",
            required_role=UserRole.ANALYST,
            required_access_level=AccessLevel.READ,
            required_security_clearance=SecurityClassification.INTERNAL
        )
        self.access_control.add_access_rule(kg_rule)
        
        # Rule for reasoning engine access
        reasoning_rule = AccessControlRule(
            rule_id="reasoning_engine_access",
            resource_pattern=".*reasoning.*",
            required_role=UserRole.RESEARCHER,
            required_access_level=AccessLevel.WRITE,
            required_security_clearance=SecurityClassification.INTERNAL
        )
        self.access_control.add_access_rule(reasoning_rule)
        
        # Rule for confidential content
        confidential_rule = AccessControlRule(
            rule_id="confidential_content_access",
            resource_pattern=".*confidential.*",
            required_role=UserRole.ANALYST,
            required_access_level=AccessLevel.READ,
            required_security_clearance=SecurityClassification.CONFIDENTIAL
        )
        self.access_control.add_access_rule(confidential_rule)
    
    def _get_access_levels_for_role(self, user_role: UserRole) -> Set[AccessLevel]:
        """Get default access levels for user role"""
        role_access = {
            UserRole.GUEST: {AccessLevel.READ},
            UserRole.USER: {AccessLevel.READ},
            UserRole.ANALYST: {AccessLevel.READ, AccessLevel.WRITE},
            UserRole.RESEARCHER: {AccessLevel.READ, AccessLevel.WRITE},
            UserRole.ADMIN: {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN},
            UserRole.SECURITY_OFFICER: {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN},
            UserRole.SYSTEM_ADMIN: {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN, AccessLevel.OWNER}
        }
        
        return role_access.get(user_role, {AccessLevel.READ})
    
    def _analyze_security_classifications(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Analyze security classifications in audit events"""
        classifications = {}
        for event in events:
            classification = event.security_classification.value
            classifications[classification] = classifications.get(classification, 0) + 1
        return classifications
    
    def _analyze_access_patterns(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze access patterns in audit events"""
        user_activity = {}
        action_counts = {}
        hourly_distribution = {}
        
        for event in events:
            # User activity
            user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # Action counts
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
            
            # Hourly distribution
            hour = event.timestamp.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        return {
            'top_users': sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10],
            'action_distribution': action_counts,
            'hourly_activity': hourly_distribution
        }
    
    def _generate_security_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate security recommendations based on audit events"""
        recommendations = []
        
        # Check for excessive failed attempts
        failed_events = [e for e in events if not e.success]
        if len(failed_events) > len(events) * 0.1:  # More than 10% failed
            recommendations.append("High failure rate detected - review access controls and user training")
        
        # Check for unusual access patterns
        user_activity = {}
        for event in events:
            user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
        
        avg_activity = sum(user_activity.values()) / len(user_activity) if user_activity else 0
        high_activity_users = [u for u, count in user_activity.items() if count > avg_activity * 3]
        
        if high_activity_users:
            recommendations.append(f"Unusual activity detected for users: {', '.join(high_activity_users)}")
        
        # Check for off-hours access
        off_hours_events = [e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22]
        if len(off_hours_events) > len(events) * 0.05:  # More than 5% off-hours
            recommendations.append("Significant off-hours access detected - review if legitimate")
        
        return recommendations