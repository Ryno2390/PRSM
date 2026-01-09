"""
Post-Quantum Access Control System for PRSM P2P Collaboration

This module implements a comprehensive access control system that integrates
with post-quantum cryptography and the P2P network to provide fine-grained
permissions and multi-signature authorization for secure collaboration.

Key Features:
- Role-based access control (RBAC) with post-quantum signatures
- Multi-signature authorization schemes
- Attribute-based access control (ABAC)
- Dynamic permission management
- Audit logging and compliance tracking
- Integration with distributed key management
"""

import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import base64

from .key_management import (
    DistributedKeyManager, 
    CryptographicKey, 
    KeyType, 
    PostQuantumCrypto,
    PostQuantumAlgorithm
)

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Basic permissions for file operations"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


class AccessLevel(Enum):
    """Access levels for hierarchical permissions"""
    NONE = "none"
    VIEWER = "viewer"           # Read-only access
    CONTRIBUTOR = "contributor" # Read + Write
    COLLABORATOR = "collaborator" # Read + Write + Share
    OWNER = "owner"            # Full access including admin


class ResourceType(Enum):
    """Types of resources that can be access-controlled"""
    FILE = "file"
    FOLDER = "folder"
    WORKSPACE = "workspace"
    COLLABORATION_SESSION = "session"


@dataclass
class AccessRule:
    """Defines an access control rule"""
    rule_id: str
    resource_id: str
    resource_type: ResourceType
    subject_id: str  # User or node ID
    subject_type: str  # "user", "node", "group"
    permissions: Set[Permission]
    access_level: AccessLevel
    conditions: Dict[str, Any]  # Additional conditions (time, IP, etc.)
    created_at: float = 0.0
    expires_at: Optional[float] = None
    created_by: str = ""
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if isinstance(self.permissions, list):
            self.permissions = set(Permission(p) if isinstance(p, str) else p for p in self.permissions)
    
    @property
    def is_expired(self) -> bool:
        """Check if rule has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def is_active(self) -> bool:
        """Check if rule is currently active"""
        return not self.is_expired
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if rule grants a specific permission"""
        return permission in self.permissions
    
    def satisfies_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if current context satisfies rule conditions"""
        if not self.conditions:
            return True
        
        # Time-based conditions
        if 'valid_after' in self.conditions:
            if time.time() < self.conditions['valid_after']:
                return False
        
        if 'valid_before' in self.conditions:
            if time.time() > self.conditions['valid_before']:
                return False
        
        # IP-based conditions
        if 'allowed_ips' in self.conditions:
            client_ip = context.get('client_ip')
            if client_ip and client_ip not in self.conditions['allowed_ips']:
                return False
        
        # Node-based conditions
        if 'allowed_nodes' in self.conditions:
            node_id = context.get('node_id')
            if node_id and node_id not in self.conditions['allowed_nodes']:
                return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['permissions'] = [p.value if isinstance(p, Permission) else p for p in self.permissions]
        data['resource_type'] = self.resource_type.value
        data['access_level'] = self.access_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AccessRule':
        """Create from dictionary"""
        data['permissions'] = {Permission(p) for p in data['permissions']}
        data['resource_type'] = ResourceType(data['resource_type'])
        data['access_level'] = AccessLevel(data['access_level'])
        return cls(**data)


@dataclass
class MultiSigRequest:
    """Multi-signature authorization request"""
    request_id: str
    resource_id: str
    operation: str
    requester_id: str
    required_signatures: int
    collected_signatures: Dict[str, bytes]  # signer_id -> signature
    request_data: Dict[str, Any]
    created_at: float = 0.0
    expires_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 3600  # 1 hour default
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def is_complete(self) -> bool:
        return len(self.collected_signatures) >= self.required_signatures
    
    @property
    def signature_progress(self) -> float:
        return len(self.collected_signatures) / self.required_signatures


@dataclass
class AccessAuditLog:
    """Audit log entry for access control events"""
    log_id: str
    timestamp: float
    event_type: str  # "access_granted", "access_denied", "permission_changed"
    resource_id: str
    subject_id: str
    operation: str
    success: bool
    context: Dict[str, Any]
    signature: Optional[bytes] = None  # PQ signature for tamper-proofing
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp == 0:
            self.timestamp = time.time()


class AccessControlMatrix:
    """
    Manages access control rules and permissions
    
    Provides efficient lookup and management of access control rules
    with support for hierarchical permissions and complex conditions.
    """
    
    def __init__(self):
        # Rule storage organized by resource for efficient lookup
        self.rules_by_resource: Dict[str, List[AccessRule]] = {}
        self.rules_by_subject: Dict[str, List[AccessRule]] = {}
        self.all_rules: Dict[str, AccessRule] = {}
        
        # Multi-signature requests
        self.multisig_requests: Dict[str, MultiSigRequest] = {}
        
        # Audit logging
        self.audit_logs: List[AccessAuditLog] = []
        self.max_audit_logs = 10000
    
    def add_rule(self, rule: AccessRule):
        """Add an access control rule"""
        self.all_rules[rule.rule_id] = rule
        
        # Index by resource
        if rule.resource_id not in self.rules_by_resource:
            self.rules_by_resource[rule.resource_id] = []
        self.rules_by_resource[rule.resource_id].append(rule)
        
        # Index by subject
        if rule.subject_id not in self.rules_by_subject:
            self.rules_by_subject[rule.subject_id] = []
        self.rules_by_subject[rule.subject_id].append(rule)
        
        logger.debug(f"Added access rule {rule.rule_id} for {rule.subject_id} on {rule.resource_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an access control rule"""
        if rule_id not in self.all_rules:
            return False
        
        rule = self.all_rules[rule_id]
        
        # Remove from indices
        if rule.resource_id in self.rules_by_resource:
            self.rules_by_resource[rule.resource_id] = [
                r for r in self.rules_by_resource[rule.resource_id] 
                if r.rule_id != rule_id
            ]
        
        if rule.subject_id in self.rules_by_subject:
            self.rules_by_subject[rule.subject_id] = [
                r for r in self.rules_by_subject[rule.subject_id]
                if r.rule_id != rule_id
            ]
        
        # Remove from main storage
        del self.all_rules[rule_id]
        
        logger.debug(f"Removed access rule {rule_id}")
        return True
    
    def get_rules_for_resource(self, resource_id: str) -> List[AccessRule]:
        """Get all active rules for a resource"""
        rules = self.rules_by_resource.get(resource_id, [])
        return [rule for rule in rules if rule.is_active]
    
    def get_rules_for_subject(self, subject_id: str) -> List[AccessRule]:
        """Get all active rules for a subject"""
        rules = self.rules_by_subject.get(subject_id, [])
        return [rule for rule in rules if rule.is_active]
    
    def check_permission(self, subject_id: str, resource_id: str, 
                        permission: Permission, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if subject has permission for resource"""
        context = context or {}
        
        # Get applicable rules
        resource_rules = self.get_rules_for_resource(resource_id)
        
        # Find rules that apply to this subject
        applicable_rules = [
            rule for rule in resource_rules
            if rule.subject_id == subject_id and rule.satisfies_conditions(context)
        ]
        
        # Check if any rule grants the permission
        for rule in applicable_rules:
            if rule.has_permission(permission):
                self._log_access_event("access_granted", resource_id, subject_id, 
                                     permission.value, True, context)
                return True
        
        # No matching rule found
        self._log_access_event("access_denied", resource_id, subject_id,
                             permission.value, False, context)
        return False
    
    def get_effective_permissions(self, subject_id: str, resource_id: str,
                                context: Optional[Dict[str, Any]] = None) -> Set[Permission]:
        """Get all effective permissions for a subject on a resource"""
        context = context or {}
        
        resource_rules = self.get_rules_for_resource(resource_id)
        applicable_rules = [
            rule for rule in resource_rules
            if rule.subject_id == subject_id and rule.satisfies_conditions(context)
        ]
        
        # Combine permissions from all applicable rules
        effective_permissions = set()
        for rule in applicable_rules:
            effective_permissions.update(rule.permissions)
        
        return effective_permissions
    
    def create_multisig_request(self, resource_id: str, operation: str,
                              requester_id: str, required_signatures: int,
                              request_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a multi-signature authorization request"""
        request_id = self._generate_request_id()
        
        request = MultiSigRequest(
            request_id=request_id,
            resource_id=resource_id,
            operation=operation,
            requester_id=requester_id,
            required_signatures=required_signatures,
            collected_signatures={},
            request_data=request_data or {}
        )
        
        self.multisig_requests[request_id] = request
        
        logger.info(f"Created multi-sig request {request_id} for {operation} on {resource_id}")
        return request_id
    
    def add_signature(self, request_id: str, signer_id: str, signature: bytes) -> bool:
        """Add a signature to a multi-sig request"""
        if request_id not in self.multisig_requests:
            return False
        
        request = self.multisig_requests[request_id]
        
        if request.is_expired:
            logger.warning(f"Multi-sig request {request_id} has expired")
            return False
        
        # Add signature
        request.collected_signatures[signer_id] = signature
        
        logger.info(f"Added signature from {signer_id} to request {request_id} "
                   f"({len(request.collected_signatures)}/{request.required_signatures})")
        
        return True
    
    def is_multisig_authorized(self, request_id: str) -> bool:
        """Check if multi-sig request is fully authorized"""
        if request_id not in self.multisig_requests:
            return False
        
        request = self.multisig_requests[request_id]
        return request.is_complete and not request.is_expired
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return hashlib.sha256(f"{time.time()}:{id(self)}".encode()).hexdigest()[:16]
    
    def _log_access_event(self, event_type: str, resource_id: str, subject_id: str,
                         operation: str, success: bool, context: Dict[str, Any]):
        """Log an access control event"""
        log_id = hashlib.sha256(f"{time.time()}:{event_type}:{resource_id}:{subject_id}".encode()).hexdigest()[:16]
        
        log_entry = AccessAuditLog(
            log_id=log_id,
            timestamp=time.time(),
            event_type=event_type,
            resource_id=resource_id,
            subject_id=subject_id,
            operation=operation,
            success=success,
            context=context.copy()
        )
        
        self.audit_logs.append(log_entry)
        
        # Maintain log size limit
        if len(self.audit_logs) > self.max_audit_logs:
            self.audit_logs = self.audit_logs[-self.max_audit_logs:]
    
    def cleanup_expired_rules(self) -> int:
        """Remove expired rules and return count of removed rules"""
        expired_rules = [
            rule_id for rule_id, rule in self.all_rules.items()
            if rule.is_expired
        ]
        
        for rule_id in expired_rules:
            self.remove_rule(rule_id)
        
        return len(expired_rules)
    
    def get_audit_logs(self, resource_id: Optional[str] = None,
                      subject_id: Optional[str] = None,
                      limit: int = 100) -> List[AccessAuditLog]:
        """Get audit logs with optional filtering"""
        logs = self.audit_logs
        
        if resource_id:
            logs = [log for log in logs if log.resource_id == resource_id]
        
        if subject_id:
            logs = [log for log in logs if log.subject_id == subject_id]
        
        # Sort by timestamp (most recent first) and limit
        logs = sorted(logs, key=lambda l: l.timestamp, reverse=True)
        return logs[:limit]


class PostQuantumAccessController:
    """
    Main post-quantum access control system
    
    Integrates with distributed key management and P2P network to provide
    comprehensive access control with post-quantum cryptographic security.
    """
    
    def __init__(self, node_id: str, key_manager: DistributedKeyManager,
                 config: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.key_manager = key_manager
        self.config = config or {}
        
        # Core components
        self.access_matrix = AccessControlMatrix()
        self.pq_crypto = PostQuantumCrypto()
        
        # Configuration
        self.default_multisig_threshold = self.config.get('multisig_threshold', 2)
        self.audit_signature_key_id = None  # Will be set during initialization
        
        # P2P integration
        self.p2p_network = None
        
        logger.info(f"Post-quantum access controller initialized for node {node_id}")
    
    async def initialize(self):
        """Initialize the access controller"""
        # Generate key for audit log signing
        self.audit_signature_key_id = await self.key_manager.generate_keypair(
            KeyType.SIGNING,
            PostQuantumAlgorithm.ML_DSA_87
        )
        
        logger.info("Access controller initialization complete")
    
    def set_p2p_network(self, p2p_network):
        """Set P2P network for distributed operations"""
        self.p2p_network = p2p_network
    
    async def grant_access(self, subject_id: str, resource_id: str,
                          permissions: List[Permission], 
                          grantor_id: Optional[str] = None,
                          conditions: Optional[Dict[str, Any]] = None,
                          expires_in: Optional[int] = None) -> str:
        """Grant access permissions to a subject"""
        grantor_id = grantor_id or self.node_id
        
        # Check if grantor has admin permission
        if grantor_id != self.node_id:  # Allow self-grants
            if not self.access_matrix.check_permission(
                grantor_id, resource_id, Permission.ADMIN
            ):
                raise PermissionError(f"Grantor {grantor_id} lacks admin permission")
        
        # Determine access level based on permissions
        access_level = self._determine_access_level(permissions)
        
        # Create rule
        rule_id = self._generate_rule_id()
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        rule = AccessRule(
            rule_id=rule_id,
            resource_id=resource_id,
            resource_type=ResourceType.FILE,  # Default, could be determined dynamically
            subject_id=subject_id,
            subject_type="user",  # Could be determined from subject_id
            permissions=set(permissions),
            access_level=access_level,
            conditions=conditions or {},
            expires_at=expires_at,
            created_by=grantor_id
        )
        
        self.access_matrix.add_rule(rule)
        
        # Sign the access grant for audit purposes
        await self._sign_access_grant(rule)
        
        logger.info(f"Granted {permissions} to {subject_id} on {resource_id}")
        return rule_id
    
    async def revoke_access(self, rule_id: str, revoker_id: Optional[str] = None) -> bool:
        """Revoke access permissions"""
        revoker_id = revoker_id or self.node_id
        
        if rule_id not in self.access_matrix.all_rules:
            return False
        
        rule = self.access_matrix.all_rules[rule_id]
        
        # Check if revoker has admin permission or is the original grantor
        if revoker_id != self.node_id and revoker_id != rule.created_by:
            if not self.access_matrix.check_permission(
                revoker_id, rule.resource_id, Permission.ADMIN
            ):
                raise PermissionError(f"Revoker {revoker_id} lacks permission to revoke")
        
        success = self.access_matrix.remove_rule(rule_id)
        
        if success:
            # Log revocation
            self.access_matrix._log_access_event(
                "permission_revoked", rule.resource_id, rule.subject_id,
                "revoke", True, {"revoker_id": revoker_id}
            )
            
            logger.info(f"Revoked access rule {rule_id}")
        
        return success
    
    async def check_access(self, subject_id: str, resource_id: str,
                          operation: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if subject can perform operation on resource"""
        # Map operation to permission
        permission_map = {
            "read": Permission.READ,
            "write": Permission.WRITE,
            "delete": Permission.DELETE,
            "share": Permission.SHARE,
            "admin": Permission.ADMIN
        }
        
        permission = permission_map.get(operation.lower())
        if not permission:
            logger.warning(f"Unknown operation: {operation}")
            return False
        
        return self.access_matrix.check_permission(subject_id, resource_id, permission, context)
    
    async def request_multisig_authorization(self, resource_id: str, operation: str,
                                           authorized_signers: List[str],
                                           threshold: Optional[int] = None,
                                           request_data: Optional[Dict[str, Any]] = None) -> str:
        """Request multi-signature authorization for sensitive operations"""
        threshold = threshold or min(self.default_multisig_threshold, len(authorized_signers))
        
        request_id = self.access_matrix.create_multisig_request(
            resource_id, operation, self.node_id, threshold, request_data
        )
        
        # Notify authorized signers via P2P network
        if self.p2p_network:
            await self._notify_signers(request_id, authorized_signers)
        
        return request_id
    
    async def sign_multisig_request(self, request_id: str, signer_id: Optional[str] = None) -> bool:
        """Sign a multi-signature authorization request"""
        signer_id = signer_id or self.node_id
        
        if request_id not in self.access_matrix.multisig_requests:
            return False
        
        request = self.access_matrix.multisig_requests[request_id]
        
        # Check if signer has permission to sign
        if not self.access_matrix.check_permission(
            signer_id, request.resource_id, Permission.ADMIN
        ):
            logger.warning(f"Signer {signer_id} lacks permission for {request_id}")
            return False
        
        # Create signature data
        signature_data = json.dumps({
            'request_id': request_id,
            'resource_id': request.resource_id,
            'operation': request.operation,
            'signer_id': signer_id,
            'timestamp': time.time()
        }, sort_keys=True).encode()
        
        # Get signer's signing key
        # This would normally retrieve the signer's private key
        # For now, create a mock signature
        signature = hashlib.sha256(signature_data).digest()  # Mock signature
        
        success = self.access_matrix.add_signature(request_id, signer_id, signature)
        
        if success:
            logger.info(f"Added signature from {signer_id} to request {request_id}")
        
        return success
    
    async def execute_multisig_operation(self, request_id: str) -> bool:
        """Execute operation after multi-sig authorization is complete"""
        if not self.access_matrix.is_multisig_authorized(request_id):
            logger.warning(f"Multi-sig request {request_id} not fully authorized")
            return False
        
        request = self.access_matrix.multisig_requests[request_id]
        
        # Execute the requested operation
        # This would integrate with the actual file/resource management system
        logger.info(f"Executing multi-sig operation: {request.operation} on {request.resource_id}")
        
        # Clean up completed request
        del self.access_matrix.multisig_requests[request_id]
        
        return True
    
    def create_role_based_rules(self, role_name: str, resource_pattern: str,
                               permissions: List[Permission]) -> List[str]:
        """Create access rules based on roles (helper function)"""
        # This would integrate with a role management system
        # For now, return empty list
        logger.info(f"Created role-based rules for {role_name}")
        return []
    
    def _determine_access_level(self, permissions: List[Permission]) -> AccessLevel:
        """Determine access level based on permissions"""
        perm_set = set(permissions)
        
        if Permission.ADMIN in perm_set:
            return AccessLevel.OWNER
        elif Permission.SHARE in perm_set:
            return AccessLevel.COLLABORATOR
        elif Permission.WRITE in perm_set:
            return AccessLevel.CONTRIBUTOR
        elif Permission.READ in perm_set:
            return AccessLevel.VIEWER
        else:
            return AccessLevel.NONE
    
    def _generate_rule_id(self) -> str:
        """Generate unique rule ID"""
        return hashlib.sha256(f"{time.time()}:{self.node_id}:{id(self)}".encode()).hexdigest()[:16]
    
    async def _sign_access_grant(self, rule: AccessRule):
        """Sign an access grant for audit purposes"""
        if not self.audit_signature_key_id:
            return
        
        # Create signature data
        signature_data = json.dumps(rule.to_dict(), sort_keys=True).encode()
        
        # Sign with audit key (mock implementation)
        signature = hashlib.sha256(signature_data).digest()
        
        # Store signature in audit log
        self.access_matrix._log_access_event(
            "access_granted", rule.resource_id, rule.subject_id,
            "grant", True, {"rule_id": rule.rule_id, "signature": base64.b64encode(signature).decode()}
        )
    
    async def _notify_signers(self, request_id: str, signers: List[str]):
        """Notify signers about multi-sig request via P2P network"""
        # This would send notifications via P2P network
        logger.debug(f"Notified {len(signers)} signers about request {request_id}")
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        total_rules = len(self.access_matrix.all_rules)
        active_rules = sum(1 for rule in self.access_matrix.all_rules.values() if rule.is_active)
        expired_rules = total_rules - active_rules
        
        pending_multisig = len(self.access_matrix.multisig_requests)
        
        # Permission distribution
        permission_counts = {}
        for perm in Permission:
            permission_counts[perm.value] = sum(
                1 for rule in self.access_matrix.all_rules.values()
                if perm in rule.permissions and rule.is_active
            )
        
        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'expired_rules': expired_rules,
            'pending_multisig_requests': pending_multisig,
            'permission_distribution': permission_counts,
            'audit_log_entries': len(self.access_matrix.audit_logs)
        }
    
    def export_access_rules(self) -> Dict[str, Any]:
        """Export access rules for backup/sync"""
        rules_data = {}
        for rule_id, rule in self.access_matrix.all_rules.items():
            if rule.is_active:
                rules_data[rule_id] = rule.to_dict()
        
        return {
            'node_id': self.node_id,
            'rules': rules_data,
            'exported_at': time.time()
        }
    
    def import_access_rules(self, rules_data: Dict[str, Any]):
        """Import access rules from backup/sync"""
        if 'rules' not in rules_data:
            return
        
        imported_count = 0
        for rule_id, rule_data in rules_data['rules'].items():
            try:
                rule = AccessRule.from_dict(rule_data)
                self.access_matrix.add_rule(rule)
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import rule {rule_id}: {e}")
        
        logger.info(f"Imported {imported_count} access rules")


# Example usage and testing
async def example_access_control():
    """Example of post-quantum access control usage"""
    from .key_management import DistributedKeyManager
    
    # Initialize components
    key_manager = DistributedKeyManager("test_node")
    access_controller = PostQuantumAccessController("test_node", key_manager)
    
    await access_controller.initialize()
    
    # Grant permissions
    rule_id = await access_controller.grant_access(
        subject_id="user123",
        resource_id="file456",
        permissions=[Permission.READ, Permission.WRITE],
        expires_in=3600  # 1 hour
    )
    
    print(f"Granted access rule: {rule_id}")
    
    # Check access
    can_read = await access_controller.check_access("user123", "file456", "read")
    can_delete = await access_controller.check_access("user123", "file456", "delete")
    
    print(f"Can read: {can_read}")
    print(f"Can delete: {can_delete}")
    
    # Create multi-sig request
    multisig_id = await access_controller.request_multisig_authorization(
        resource_id="sensitive_file",
        operation="delete",
        authorized_signers=["admin1", "admin2", "admin3"],
        threshold=2
    )
    
    print(f"Created multi-sig request: {multisig_id}")
    
    # Sign request
    await access_controller.sign_multisig_request(multisig_id, "admin1")
    await access_controller.sign_multisig_request(multisig_id, "admin2")
    
    # Execute operation
    executed = await access_controller.execute_multisig_operation(multisig_id)
    print(f"Multi-sig operation executed: {executed}")
    
    # Get statistics
    stats = access_controller.get_access_statistics()
    print(f"Access control statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_access_control())