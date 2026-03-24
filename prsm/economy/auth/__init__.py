"""Re-export shim: delegates to prsm.core.auth"""
from prsm.core.auth.auth_manager import auth_manager, get_current_user  # noqa: F401
from prsm.core.auth.models import User, UserRole  # noqa: F401
