#!/usr/bin/env python3
"""
Authentication Integration Script
Adds authentication system to the main PRSM API
"""

import re
from pathlib import Path

def integrate_auth_to_main_api():
    """Add authentication integration to main.py"""
    
    main_api_path = Path("prsm/api/main.py")
    
    if not main_api_path.exists():
        print(f"❌ {main_api_path} not found")
        return False
    
    # Read current main.py
    with open(main_api_path, 'r') as f:
        content = f.read()
    
    # Check if auth is already integrated
    if "from prsm.auth" in content:
        print("✅ Authentication already integrated in main.py")
        return True
    
    # Add auth imports after existing imports
    import_section = """from prsm.api.teams_api import router as teams_router"""
    
    auth_imports = """from prsm.api.teams_api import router as teams_router
from prsm.api.auth_api import router as auth_router
from prsm.auth.auth_manager import auth_manager
from prsm.auth.middleware import AuthMiddleware, SecurityHeadersMiddleware"""
    
    content = content.replace(import_section, auth_imports)
    
    # Find the FastAPI app creation and add middleware
    app_creation_pattern = r'(app = FastAPI\([^)]+\))'
    
    if re.search(app_creation_pattern, content):
        # Add middleware after app creation
        middleware_addition = '''
# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuthMiddleware, rate_limit_requests=100, rate_limit_window=60)

# Initialize auth manager
@app.on_event("startup")
async def initialize_auth():
    """Initialize authentication system"""
    try:
        await auth_manager.initialize()
        logger.info("Authentication system initialized")
    except Exception as e:
        logger.error("Failed to initialize auth system", error=str(e))

# Include authentication router
app.include_router(auth_router)
'''
        
        # Find the teams router inclusion and add auth router
        teams_router_pattern = r'(app\.include_router\(teams_router\))'
        
        if re.search(teams_router_pattern, content):
            content = re.sub(teams_router_pattern, 
                           r'\1\napp.include_router(auth_router)', 
                           content)
        else:
            # Add auth router after app creation
            content = re.sub(app_creation_pattern, 
                           r'\1' + middleware_addition, 
                           content)
    
    # Write updated content
    with open(main_api_path, 'w') as f:
        f.write(content)
    
    print("✅ Authentication integration added to main.py")
    return True

def create_auth_requirements():
    """Add authentication dependencies to requirements"""
    
    requirements_path = Path("requirements.txt")
    
    if not requirements_path.exists():
        print(f"❌ {requirements_path} not found")
        return False
    
    # Read current requirements
    with open(requirements_path, 'r') as f:
        requirements = f.read()
    
    # Auth dependencies to add
    auth_deps = """
# Authentication dependencies
PyJWT==2.8.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
"""
    
    # Check if auth deps already exist
    if "PyJWT" in requirements:
        print("✅ Authentication dependencies already in requirements.txt")
        return True
    
    # Add auth dependencies
    with open(requirements_path, 'a') as f:
        f.write(auth_deps)
    
    print("✅ Authentication dependencies added to requirements.txt")
    return True

def create_auth_config():
    """Add authentication configuration to config.py"""
    
    config_path = Path("prsm/core/config.py")
    
    if not config_path.exists():
        print(f"❌ {config_path} not found")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check if auth config already exists
    if "jwt_algorithm" in content:
        print("✅ Authentication config already in config.py")
        return True
    
    # Add auth configuration before the class end
    auth_config = '''    
    # === Authentication Configuration ===
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30  # Access token expiry
    jwt_refresh_expire_days: int = 7  # Refresh token expiry
    
    # Password policy
    password_min_length: int = 8
    max_login_attempts: int = 5
    account_lockout_minutes: int = 15
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Security headers
    enable_security_headers: bool = True
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "https://prsm.ai"], 
        env="PRSM_CORS_ORIGINS"
    )'''
    
    # Find the end of the PRSMSettings class and add auth config
    class_end_pattern = r'(\s+def validate_required_config\(self\))'
    
    if re.search(class_end_pattern, content):
        content = re.sub(class_end_pattern, auth_config + r'\1', content)
    else:
        # Fallback: add before the last method
        content = content.replace(
            '    @property\n    def is_production(self) -> bool:',
            auth_config + '\n    @property\n    def is_production(self) -> bool:'
        )
    
    # Write updated content
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✅ Authentication configuration added to config.py")
    return True

def main():
    """Main integration function"""
    print("🔒 Integrating PRSM Authentication System...")
    
    success = True
    
    # Step 1: Add auth requirements
    if not create_auth_requirements():
        success = False
    
    # Step 2: Add auth config
    if not create_auth_config():
        success = False
    
    # Step 3: Integrate auth to main API
    if not integrate_auth_to_main_api():
        success = False
    
    if success:
        print("\n🎉 Authentication Integration Complete!")
        print("\n📋 Next Steps:")
        print("1. Install new dependencies: pip install -r requirements.txt")
        print("2. Update environment variables with JWT_SECRET_KEY")
        print("3. Run database migrations for auth tables")
        print("4. Test authentication endpoints")
        print("\n🔗 Available Auth Endpoints:")
        print("- POST /auth/register - User registration")
        print("- POST /auth/login - User login")
        print("- POST /auth/refresh - Token refresh")
        print("- POST /auth/logout - User logout")
        print("- GET /auth/me - Current user profile")
        print("- GET /auth/health - Auth system health")
    else:
        print("\n❌ Authentication integration failed!")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    main()