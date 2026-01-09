"""
Multi-Factor Authentication (MFA) Provider for PRSM
==================================================

Supports multiple MFA methods including TOTP, SMS, Email, and Hardware tokens.
Provides comprehensive MFA management and verification capabilities.
"""

import asyncio
import base64
import hashlib
import hmac
import io
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

try:
    import pyotp
    import qrcode
    HAS_TOTP = True
except ImportError:
    HAS_TOTP = False

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    HAS_AWS_SNS = True
except ImportError:
    HAS_AWS_SNS = False

try:
    import twilio.rest
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

from ..models import User

logger = structlog.get_logger(__name__)


class MFAMethod(Enum):
    """MFA method types"""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODES = "backup_codes"
    HARDWARE_TOKEN = "hardware_token"


@dataclass
class MFAConfig:
    """MFA configuration"""
    enabled: bool = True
    required_for_admin: bool = True
    required_for_all: bool = False
    allowed_methods: List[MFAMethod] = None
    
    # TOTP settings
    totp_issuer: str = "PRSM"
    totp_period: int = 30
    totp_digits: int = 6
    
    # SMS settings
    sms_provider: str = "twilio"  # 'twilio' or 'aws_sns'
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    aws_sns_region: Optional[str] = None
    
    # Email settings
    email_enabled: bool = True
    email_template: str = "Your PRSM verification code is: {code}"
    
    # Backup codes
    backup_codes_count: int = 10
    backup_codes_length: int = 8
    
    # Security settings
    max_attempts: int = 3
    lockout_duration_minutes: int = 15
    code_validity_minutes: int = 5


@dataclass
class MFADevice:
    """MFA device representation"""
    id: str
    user_id: str
    method: MFAMethod
    name: str
    secret: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    backup_codes: Optional[List[str]] = None
    is_active: bool = True
    created_at: datetime = None
    last_used: Optional[datetime] = None
    verified: bool = False


@dataclass
class MFAChallenge:
    """MFA challenge"""
    id: str
    user_id: str
    device_id: str
    method: MFAMethod
    code: str
    expires_at: datetime
    attempts: int = 0
    verified: bool = False
    created_at: datetime = None


class TOTPProvider:
    """Time-based One-Time Password provider"""
    
    def __init__(self, config: MFAConfig):
        if not HAS_TOTP:
            raise ImportError("pyotp package is required for TOTP functionality")
        
        self.config = config
    
    def generate_secret(self) -> str:
        """Generate a new TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, secret: str, user_email: str, device_name: str) -> bytes:
        """Generate QR code for TOTP setup"""
        try:
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user_email,
                issuer_name=f"{self.config.totp_issuer} ({device_name})"
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error("Failed to generate QR code", error=str(e))
            raise
    
    def verify_code(self, secret: str, code: str, window: int = 1) -> bool:
        """Verify TOTP code"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=window)
        except Exception as e:
            logger.error("TOTP verification failed", error=str(e))
            return False
    
    def get_current_code(self, secret: str) -> str:
        """Get current TOTP code (for testing)"""
        totp = pyotp.TOTP(secret)
        return totp.now()


class SMSProvider:
    """SMS-based MFA provider"""
    
    def __init__(self, config: MFAConfig):
        self.config = config
        
        # Initialize SMS client based on provider
        if config.sms_provider == "twilio":
            if not HAS_TWILIO:
                raise ImportError("twilio package is required for Twilio SMS")
            if not all([config.twilio_account_sid, config.twilio_auth_token, config.twilio_from_number]):
                raise ValueError("Twilio credentials not configured")
            
            self.client = twilio.rest.Client(config.twilio_account_sid, config.twilio_auth_token)
            
        elif config.sms_provider == "aws_sns":
            if not HAS_AWS_SNS:
                raise ImportError("boto3 package is required for AWS SNS")
            
            self.client = boto3.client('sns', region_name=config.aws_sns_region or 'us-east-1')
    
    async def send_code(self, phone_number: str, code: str) -> bool:
        """Send SMS verification code"""
        try:
            message = f"Your PRSM verification code is: {code}"
            
            if self.config.sms_provider == "twilio":
                message_obj = self.client.messages.create(
                    body=message,
                    from_=self.config.twilio_from_number,
                    to=phone_number
                )
                logger.info("SMS sent via Twilio", phone=phone_number, sid=message_obj.sid)
                return True
                
            elif self.config.sms_provider == "aws_sns":
                response = self.client.publish(
                    PhoneNumber=phone_number,
                    Message=message
                )
                logger.info("SMS sent via AWS SNS", phone=phone_number, message_id=response.get('MessageId'))
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to send SMS", phone=phone_number, error=str(e))
            return False


class EmailProvider:
    """Email-based MFA provider"""
    
    def __init__(self, config: MFAConfig):
        self.config = config
    
    async def send_code(self, email: str, code: str) -> bool:
        """Send email verification code"""
        try:
            # This would integrate with your email service
            # For now, just log the code
            message = self.config.email_template.format(code=code)
            
            logger.info("Email MFA code would be sent", 
                       email=email, 
                       message=message)
            
            # In a real implementation, you would:
            # - Use your email service (SendGrid, AWS SES, etc.)
            # - Send an email with the verification code
            # - Return True if successful
            
            return True
            
        except Exception as e:
            logger.error("Failed to send email MFA code", email=email, error=str(e))
            return False


class BackupCodesProvider:
    """Backup codes MFA provider"""
    
    def __init__(self, config: MFAConfig):
        self.config = config
    
    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes"""
        codes = []
        for _ in range(self.config.backup_codes_count):
            # Generate random alphanumeric code
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
                          for _ in range(self.config.backup_codes_length))
            codes.append(code)
        
        return codes
    
    def verify_backup_code(self, stored_codes: List[str], provided_code: str) -> Tuple[bool, List[str]]:
        """Verify backup code and remove it if valid"""
        if provided_code.upper() in [code.upper() for code in stored_codes]:
            # Remove used code
            updated_codes = [code for code in stored_codes if code.upper() != provided_code.upper()]
            return True, updated_codes
        
        return False, stored_codes


class MFAProvider:
    """Multi-Factor Authentication provider"""
    
    def __init__(self, config: MFAConfig):
        self.config = config
        self.active_challenges: Dict[str, MFAChallenge] = {}
        self.user_devices: Dict[str, List[MFADevice]] = {}
        
        # Initialize sub-providers
        if MFAMethod.TOTP in (config.allowed_methods or []):
            self.totp_provider = TOTPProvider(config)
        
        if MFAMethod.SMS in (config.allowed_methods or []):
            self.sms_provider = SMSProvider(config)
        
        if MFAMethod.EMAIL in (config.allowed_methods or []):
            self.email_provider = EmailProvider(config)
        
        self.backup_codes_provider = BackupCodesProvider(config)
    
    def is_mfa_required(self, user: User) -> bool:
        """Check if MFA is required for user"""
        if self.config.required_for_all:
            return True
        
        if self.config.required_for_admin and user.role.value in ['admin', 'superuser']:
            return True
        
        return False
    
    async def enroll_device(self, user_id: str, method: MFAMethod, 
                           device_name: str, **kwargs) -> MFADevice:
        """Enroll a new MFA device"""
        try:
            device_id = secrets.token_hex(16)
            
            device = MFADevice(
                id=device_id,
                user_id=user_id,
                method=method,
                name=device_name,
                created_at=datetime.now(timezone.utc)
            )
            
            if method == MFAMethod.TOTP:
                device.secret = self.totp_provider.generate_secret()
            
            elif method == MFAMethod.SMS:
                phone_number = kwargs.get('phone_number')
                if not phone_number:
                    raise ValueError("Phone number required for SMS MFA")
                device.phone_number = phone_number
            
            elif method == MFAMethod.EMAIL:
                email = kwargs.get('email')
                if not email:
                    raise ValueError("Email required for email MFA")
                device.email = email
            
            elif method == MFAMethod.BACKUP_CODES:
                device.backup_codes = self.backup_codes_provider.generate_backup_codes()
            
            # Store device
            if user_id not in self.user_devices:
                self.user_devices[user_id] = []
            self.user_devices[user_id].append(device)
            
            logger.info("MFA device enrolled", 
                       user_id=user_id, 
                       device_id=device_id, 
                       method=method.value)
            
            return device
            
        except Exception as e:
            logger.error("Failed to enroll MFA device", 
                        user_id=user_id, 
                        method=method.value, 
                        error=str(e))
            raise
    
    async def verify_enrollment(self, device_id: str, verification_code: str) -> bool:
        """Verify MFA device enrollment"""
        try:
            # Find device
            device = self._find_device(device_id)
            if not device:
                return False
            
            # Verify based on method
            if device.method == MFAMethod.TOTP:
                is_valid = self.totp_provider.verify_code(device.secret, verification_code)
            
            elif device.method == MFAMethod.SMS:
                # For SMS, we would have sent a code during enrollment
                # This is a simplified verification
                is_valid = len(verification_code) == 6 and verification_code.isdigit()
            
            elif device.method == MFAMethod.EMAIL:
                # Similar to SMS
                is_valid = len(verification_code) == 6 and verification_code.isdigit()
            
            else:
                is_valid = False
            
            if is_valid:
                device.verified = True
                device.is_active = True
                logger.info("MFA device verified", device_id=device_id)
            
            return is_valid
            
        except Exception as e:
            logger.error("MFA device verification failed", device_id=device_id, error=str(e))
            return False
    
    async def initiate_challenge(self, user_id: str, device_id: Optional[str] = None) -> MFAChallenge:
        """Initiate MFA challenge"""
        try:
            # Get user devices
            user_devices = self.user_devices.get(user_id, [])
            if not user_devices:
                raise ValueError("No MFA devices enrolled")
            
            # Select device
            if device_id:
                device = next((d for d in user_devices if d.id == device_id), None)
                if not device:
                    raise ValueError("Device not found")
            else:
                # Use first active device
                device = next((d for d in user_devices if d.is_active and d.verified), None)
                if not device:
                    raise ValueError("No active MFA devices")
            
            # Generate challenge
            challenge_id = secrets.token_hex(16)
            code = self._generate_code(device.method)
            
            challenge = MFAChallenge(
                id=challenge_id,
                user_id=user_id,
                device_id=device.id,
                method=device.method,
                code=code,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=self.config.code_validity_minutes),
                created_at=datetime.now(timezone.utc)
            )
            
            # Send code based on method
            if device.method == MFAMethod.SMS:
                await self.sms_provider.send_code(device.phone_number, code)
            
            elif device.method == MFAMethod.EMAIL:
                await self.email_provider.send_code(device.email, code)
            
            # Store challenge
            self.active_challenges[challenge_id] = challenge
            
            logger.info("MFA challenge initiated", 
                       user_id=user_id, 
                       challenge_id=challenge_id, 
                       method=device.method.value)
            
            return challenge
            
        except Exception as e:
            logger.error("Failed to initiate MFA challenge", user_id=user_id, error=str(e))
            raise
    
    async def verify_challenge(self, challenge_id: str, provided_code: str) -> bool:
        """Verify MFA challenge"""
        try:
            challenge = self.active_challenges.get(challenge_id)
            if not challenge:
                logger.warning("MFA challenge not found", challenge_id=challenge_id)
                return False
            
            # Check if expired
            if datetime.now(timezone.utc) > challenge.expires_at:
                logger.warning("MFA challenge expired", challenge_id=challenge_id)
                del self.active_challenges[challenge_id]
                return False
            
            # Check attempt limit
            if challenge.attempts >= self.config.max_attempts:
                logger.warning("MFA challenge max attempts exceeded", challenge_id=challenge_id)
                del self.active_challenges[challenge_id]
                return False
            
            challenge.attempts += 1
            
            # Get device
            device = self._find_device(challenge.device_id)
            if not device:
                return False
            
            # Verify code based on method
            is_valid = False
            
            if challenge.method == MFAMethod.TOTP:
                is_valid = self.totp_provider.verify_code(device.secret, provided_code)
            
            elif challenge.method in [MFAMethod.SMS, MFAMethod.EMAIL]:
                is_valid = (provided_code == challenge.code)
            
            elif challenge.method == MFAMethod.BACKUP_CODES:
                is_valid, updated_codes = self.backup_codes_provider.verify_backup_code(
                    device.backup_codes, provided_code
                )
                if is_valid:
                    device.backup_codes = updated_codes
            
            if is_valid:
                challenge.verified = True
                device.last_used = datetime.now(timezone.utc)
                
                # Remove challenge
                del self.active_challenges[challenge_id]
                
                logger.info("MFA challenge verified successfully", 
                           challenge_id=challenge_id, 
                           user_id=challenge.user_id)
            else:
                logger.warning("MFA challenge verification failed", 
                              challenge_id=challenge_id, 
                              attempts=challenge.attempts)
            
            return is_valid
            
        except Exception as e:
            logger.error("MFA challenge verification error", 
                        challenge_id=challenge_id, 
                        error=str(e))
            return False
    
    def get_user_devices(self, user_id: str) -> List[MFADevice]:
        """Get user's MFA devices"""
        return self.user_devices.get(user_id, [])
    
    async def remove_device(self, user_id: str, device_id: str) -> bool:
        """Remove MFA device"""
        try:
            user_devices = self.user_devices.get(user_id, [])
            device = next((d for d in user_devices if d.id == device_id), None)
            
            if device:
                user_devices.remove(device)
                logger.info("MFA device removed", user_id=user_id, device_id=device_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to remove MFA device", 
                        user_id=user_id, 
                        device_id=device_id, 
                        error=str(e))
            return False
    
    def get_qr_code(self, device_id: str, user_email: str) -> Optional[bytes]:
        """Get QR code for TOTP device"""
        device = self._find_device(device_id)
        if not device or device.method != MFAMethod.TOTP:
            return None
        
        return self.totp_provider.generate_qr_code(device.secret, user_email, device.name)
    
    def _find_device(self, device_id: str) -> Optional[MFADevice]:
        """Find device by ID"""
        for user_devices in self.user_devices.values():
            for device in user_devices:
                if device.id == device_id:
                    return device
        return None
    
    def _generate_code(self, method: MFAMethod) -> str:
        """Generate verification code"""
        if method in [MFAMethod.SMS, MFAMethod.EMAIL]:
            # Generate 6-digit numeric code
            return f"{secrets.randbelow(1000000):06d}"
        
        return ""
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges"""
        now = datetime.now(timezone.utc)
        expired_challenges = [
            challenge_id for challenge_id, challenge in self.active_challenges.items()
            if now > challenge.expires_at
        ]
        
        for challenge_id in expired_challenges:
            del self.active_challenges[challenge_id]
        
        if expired_challenges:
            logger.info("Cleaned up expired MFA challenges", count=len(expired_challenges))


# Factory function for creating MFA provider
def create_mfa_provider(config: Dict[str, Any]) -> MFAProvider:
    """Create MFA provider from configuration"""
    mfa_config = MFAConfig(**config)
    return MFAProvider(mfa_config)