"""
Enterprise LDAP/Active Directory Provider for PRSM
=================================================

Provides LDAP and Active Directory authentication and user synchronization.
Supports group-based role mapping and automatic user provisioning.
"""

import asyncio
import ssl
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
import structlog

try:
    import ldap3
    from ldap3 import Server, Connection, ALL, SUBTREE, MODIFY_REPLACE
    from ldap3.core.exceptions import LDAPException, LDAPInvalidCredentialsError
    HAS_LDAP3 = True
except ImportError:
    HAS_LDAP3 = False

from ..models import User, UserRole

logger = structlog.get_logger(__name__)


@dataclass
class LDAPConfig:
    """LDAP configuration"""
    server_uri: str
    bind_dn: str
    bind_password: str
    user_base_dn: str
    user_filter: str = "(objectClass=person)"
    user_search_scope: str = "SUBTREE"
    group_base_dn: Optional[str] = None
    group_filter: str = "(objectClass=group)"
    group_search_scope: str = "SUBTREE"
    use_ssl: bool = True
    use_tls: bool = False
    verify_ssl: bool = True
    timeout: int = 30
    page_size: int = 1000
    
    # Attribute mappings
    username_attribute: str = "sAMAccountName"
    email_attribute: str = "mail"
    first_name_attribute: str = "givenName"
    last_name_attribute: str = "sn"
    full_name_attribute: str = "displayName"
    group_membership_attribute: str = "memberOf"
    
    # Role mappings
    admin_groups: List[str] = None
    moderator_groups: List[str] = None
    default_role: UserRole = UserRole.USER
    
    # Sync settings
    auto_sync_enabled: bool = True
    sync_interval_hours: int = 24
    auto_provision: bool = True
    enabled: bool = True


@dataclass
class LDAPUser:
    """LDAP user representation"""
    dn: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    groups: List[str] = None
    attributes: Dict[str, Any] = None
    is_active: bool = True


class LDAPProvider:
    """LDAP/Active Directory provider"""
    
    def __init__(self, config: LDAPConfig):
        if not HAS_LDAP3:
            raise ImportError("ldap3 package is required for LDAP functionality")
        
        self.config = config
        self.server = None
        self.connection = None
        self.last_sync = None
        
        # Initialize LDAP server
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize LDAP server configuration"""
        try:
            # Configure SSL/TLS
            tls_config = None
            if self.config.use_ssl or self.config.use_tls:
                if self.config.verify_ssl:
                    tls_config = ldap3.Tls(
                        validate=ssl.CERT_REQUIRED,
                        version=ssl.PROTOCOL_TLS,
                        ciphers='HIGH:!aNULL:!eNULL:!MD5'
                    )
                else:
                    tls_config = ldap3.Tls(validate=ssl.CERT_NONE)
            
            # Create server object
            self.server = Server(
                self.config.server_uri,
                get_info=ALL,
                use_ssl=self.config.use_ssl,
                tls=tls_config,
                connect_timeout=self.config.timeout
            )
            
            logger.info("LDAP server initialized", server=self.config.server_uri)
            
        except Exception as e:
            logger.error("Failed to initialize LDAP server", error=str(e))
            raise
    
    async def authenticate_user(self, username: str, password: str) -> Optional[LDAPUser]:
        """Authenticate user against LDAP"""
        try:
            # Find user DN
            user_dn = await self._find_user_dn(username)
            if not user_dn:
                logger.warning("User not found in LDAP", username=username)
                return None
            
            # Attempt to bind with user credentials
            user_connection = Connection(
                self.server,
                user=user_dn,
                password=password,
                auto_bind=True,
                authentication=ldap3.SIMPLE,
                check_names=True
            )
            
            if not user_connection.bound:
                logger.warning("LDAP authentication failed", username=username)
                return None
            
            # Get user details
            ldap_user = await self._get_user_details(user_dn)
            user_connection.unbind()
            
            logger.info("LDAP authentication successful", username=username)
            return ldap_user
            
        except LDAPInvalidCredentialsError:
            logger.warning("Invalid LDAP credentials", username=username)
            return None
        except Exception as e:
            logger.error("LDAP authentication error", username=username, error=str(e))
            return None
    
    async def _find_user_dn(self, username: str) -> Optional[str]:
        """Find user DN by username"""
        try:
            await self._ensure_connection()
            
            # Search for user
            search_filter = f"(&{self.config.user_filter}({self.config.username_attribute}={username}))"
            
            self.connection.search(
                search_base=self.config.user_base_dn,
                search_filter=search_filter,
                search_scope=getattr(ldap3, self.config.user_search_scope, SUBTREE),
                attributes=['dn']
            )
            
            if self.connection.entries:
                return str(self.connection.entries[0].entry_dn)
            
            return None
            
        except Exception as e:
            logger.error("Error finding user DN", username=username, error=str(e))
            return None
    
    async def _get_user_details(self, user_dn: str) -> Optional[LDAPUser]:
        """Get detailed user information"""
        try:
            await self._ensure_connection()
            
            # Attributes to retrieve
            attributes = [
                self.config.username_attribute,
                self.config.email_attribute,
                self.config.first_name_attribute,
                self.config.last_name_attribute,
                self.config.full_name_attribute,
                self.config.group_membership_attribute,
                'userAccountControl'  # For Active Directory account status
            ]
            
            # Search for user details
            self.connection.search(
                search_base=user_dn,
                search_filter='(objectClass=*)',
                search_scope=ldap3.BASE,
                attributes=attributes
            )
            
            if not self.connection.entries:
                return None
            
            entry = self.connection.entries[0]
            
            # Extract attributes
            username = self._get_attribute_value(entry, self.config.username_attribute)
            email = self._get_attribute_value(entry, self.config.email_attribute)
            first_name = self._get_attribute_value(entry, self.config.first_name_attribute)
            last_name = self._get_attribute_value(entry, self.config.last_name_attribute)
            full_name = self._get_attribute_value(entry, self.config.full_name_attribute)
            
            # Extract group memberships
            groups = []
            group_memberships = entry[self.config.group_membership_attribute].values
            if group_memberships:
                groups = [self._extract_group_name(dn) for dn in group_memberships]
            
            # Check if account is active (for Active Directory)
            is_active = True
            user_account_control = self._get_attribute_value(entry, 'userAccountControl')
            if user_account_control:
                # Check if account is disabled (bit 2)
                is_active = not (int(user_account_control) & 2)
            
            # Create LDAP user object
            ldap_user = LDAPUser(
                dn=user_dn,
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                full_name=full_name or f"{first_name or ''} {last_name or ''}".strip(),
                groups=groups,
                is_active=is_active,
                attributes=dict(entry)
            )
            
            return ldap_user
            
        except Exception as e:
            logger.error("Error getting user details", user_dn=user_dn, error=str(e))
            return None
    
    async def search_users(self, search_filter: Optional[str] = None, limit: int = 100) -> List[LDAPUser]:
        """Search for users in LDAP"""
        try:
            await self._ensure_connection()
            
            # Use custom filter or default
            filter_string = search_filter or self.config.user_filter
            
            # Attributes to retrieve
            attributes = [
                self.config.username_attribute,
                self.config.email_attribute,
                self.config.first_name_attribute,
                self.config.last_name_attribute,
                self.config.full_name_attribute,
                self.config.group_membership_attribute
            ]
            
            # Perform search with pagination
            self.connection.search(
                search_base=self.config.user_base_dn,
                search_filter=filter_string,
                search_scope=getattr(ldap3, self.config.user_search_scope, SUBTREE),
                attributes=attributes,
                paged_size=min(limit, self.config.page_size)
            )
            
            users = []
            for entry in self.connection.entries[:limit]:
                # Extract user data
                username = self._get_attribute_value(entry, self.config.username_attribute)
                email = self._get_attribute_value(entry, self.config.email_attribute)
                
                if not username or not email:
                    continue
                
                first_name = self._get_attribute_value(entry, self.config.first_name_attribute)
                last_name = self._get_attribute_value(entry, self.config.last_name_attribute)
                full_name = self._get_attribute_value(entry, self.config.full_name_attribute)
                
                # Extract groups
                groups = []
                group_memberships = entry[self.config.group_membership_attribute].values
                if group_memberships:
                    groups = [self._extract_group_name(dn) for dn in group_memberships]
                
                ldap_user = LDAPUser(
                    dn=str(entry.entry_dn),
                    username=username,
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    full_name=full_name or f"{first_name or ''} {last_name or ''}".strip(),
                    groups=groups,
                    attributes=dict(entry)
                )
                
                users.append(ldap_user)
            
            logger.info("LDAP user search completed", count=len(users))
            return users
            
        except Exception as e:
            logger.error("Error searching users", error=str(e))
            return []
    
    async def get_groups(self, search_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get groups from LDAP"""
        try:
            if not self.config.group_base_dn:
                return []
            
            await self._ensure_connection()
            
            # Use custom filter or default
            filter_string = search_filter or self.config.group_filter
            
            # Search for groups
            self.connection.search(
                search_base=self.config.group_base_dn,
                search_filter=filter_string,
                search_scope=getattr(ldap3, self.config.group_search_scope, SUBTREE),
                attributes=['cn', 'description', 'member'],
                paged_size=self.config.page_size
            )
            
            groups = []
            for entry in self.connection.entries:
                group_name = self._get_attribute_value(entry, 'cn')
                description = self._get_attribute_value(entry, 'description')
                members = entry['member'].values if entry['member'].values else []
                
                groups.append({
                    'dn': str(entry.entry_dn),
                    'name': group_name,
                    'description': description,
                    'member_count': len(members)
                })
            
            logger.info("LDAP groups retrieved", count=len(groups))
            return groups
            
        except Exception as e:
            logger.error("Error getting groups", error=str(e))
            return []
    
    def create_user_from_ldap(self, ldap_user: LDAPUser) -> User:
        """Create PRSM user from LDAP user"""
        try:
            # Determine role based on group memberships
            role = self._determine_user_role(ldap_user.groups)
            
            # Create user object
            user = User(
                email=ldap_user.email,
                username=ldap_user.username,
                full_name=ldap_user.full_name,
                is_active=ldap_user.is_active,
                is_verified=True,  # LDAP users are pre-verified
                role=role,
                ldap_dn=ldap_user.dn,
                last_ldap_sync=datetime.now(timezone.utc)
            )
            
            return user
            
        except Exception as e:
            logger.error("Error creating user from LDAP", username=ldap_user.username, error=str(e))
            raise
    
    def _determine_user_role(self, groups: List[str]) -> UserRole:
        """Determine user role based on group memberships"""
        if not groups:
            return self.config.default_role
        
        # Convert groups to lowercase for comparison
        user_groups = [group.lower() for group in groups]
        
        # Check admin groups
        admin_groups = [g.lower() for g in (self.config.admin_groups or [])]
        if any(group in admin_groups for group in user_groups):
            return UserRole.ADMIN
        
        # Check moderator groups
        moderator_groups = [g.lower() for g in (self.config.moderator_groups or [])]
        if any(group in moderator_groups for group in user_groups):
            return UserRole.MODERATOR
        
        return self.config.default_role
    
    async def _ensure_connection(self):
        """Ensure LDAP connection is established"""
        try:
            if self.connection is None or not self.connection.bound:
                self.connection = Connection(
                    self.server,
                    user=self.config.bind_dn,
                    password=self.config.bind_password,
                    auto_bind=True,
                    authentication=ldap3.SIMPLE,
                    check_names=True
                )
                
                if self.config.use_tls and not self.config.use_ssl:
                    self.connection.start_tls()
                
            return self.connection.bound
            
        except Exception as e:
            logger.error("Failed to establish LDAP connection", error=str(e))
            raise
    
    def _get_attribute_value(self, entry, attribute_name: str) -> Optional[str]:
        """Get single attribute value from LDAP entry"""
        try:
            attr = getattr(entry, attribute_name, None)
            if attr and attr.value:
                return str(attr.value)
            return None
        except Exception:
            return None
    
    def _extract_group_name(self, group_dn: str) -> str:
        """Extract group name from DN"""
        try:
            # Extract CN from DN (e.g., "CN=GroupName,OU=Groups,DC=domain,DC=com")
            parts = group_dn.split(',')
            for part in parts:
                if part.strip().startswith('CN='):
                    return part.strip()[3:]  # Remove "CN="
            return group_dn
        except Exception:
            return group_dn
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test LDAP connection and return status"""
        try:
            await self._ensure_connection()
            
            # Test search
            self.connection.search(
                search_base=self.config.user_base_dn,
                search_filter='(objectClass=*)',
                search_scope=ldap3.BASE,
                attributes=['*']
            )
            
            return {
                'success': True,
                'server': self.config.server_uri,
                'user_base_dn': self.config.user_base_dn,
                'connection_status': 'connected',
                'server_info': str(self.server.info) if self.server.info else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'server': self.config.server_uri
            }
    
    def disconnect(self):
        """Disconnect from LDAP server"""
        try:
            if self.connection and self.connection.bound:
                self.connection.unbind()
                self.connection = None
                logger.info("LDAP connection closed")
        except Exception as e:
            logger.warning("Error closing LDAP connection", error=str(e))


class LDAPSync:
    """LDAP synchronization manager"""
    
    def __init__(self, ldap_provider: LDAPProvider):
        self.ldap_provider = ldap_provider
        self.sync_in_progress = False
    
    async def sync_users(self, force: bool = False) -> Dict[str, Any]:
        """Synchronize users from LDAP"""
        if self.sync_in_progress:
            return {'error': 'Sync already in progress'}
        
        try:
            self.sync_in_progress = True
            start_time = datetime.now(timezone.utc)
            
            # Get all users from LDAP
            ldap_users = await self.ldap_provider.search_users()
            
            sync_stats = {
                'total_ldap_users': len(ldap_users),
                'users_created': 0,
                'users_updated': 0,
                'users_deactivated': 0,
                'errors': []
            }
            
            # Process each LDAP user
            for ldap_user in ldap_users:
                try:
                    # Check if user exists in PRSM database
                    # This would require database integration
                    # For now, just create user object
                    prsm_user = self.ldap_provider.create_user_from_ldap(ldap_user)
                    sync_stats['users_created'] += 1
                    
                except Exception as e:
                    sync_stats['errors'].append({
                        'user': ldap_user.username,
                        'error': str(e)
                    })
            
            # Update sync timestamp
            self.ldap_provider.last_sync = datetime.now(timezone.utc)
            
            sync_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info("LDAP sync completed", 
                       duration=sync_duration,
                       stats=sync_stats)
            
            return {
                'success': True,
                'duration_seconds': sync_duration,
                'stats': sync_stats
            }
            
        except Exception as e:
            logger.error("LDAP sync failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.sync_in_progress = False
    
    async def sync_groups(self) -> Dict[str, Any]:
        """Synchronize groups from LDAP"""
        try:
            groups = await self.ldap_provider.get_groups()
            
            # Process groups (would integrate with database)
            
            return {
                'success': True,
                'groups_found': len(groups)
            }
            
        except Exception as e:
            logger.error("LDAP group sync failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }


# Factory function for creating LDAP provider
def create_ldap_provider(config: Dict[str, Any]) -> LDAPProvider:
    """Create LDAP provider from configuration"""
    ldap_config = LDAPConfig(**config)
    return LDAPProvider(ldap_config)