"""
PRSM API Migration and Deprecation System
Comprehensive system for managing API migrations and deprecations
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from fastapi import Request, Response, FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
import json
import logging

from .versioning import APIVersion, version_registry, get_request_version
from .compatibility import compatibility_engine, migration_guide_generator

logger = logging.getLogger(__name__)


class MigrationComplexity(Enum):
    """Migration complexity levels"""
    TRIVIAL = "trivial"      # No breaking changes, automatic migration
    SIMPLE = "simple"        # Minor changes, straightforward migration
    MODERATE = "moderate"    # Some breaking changes, requires attention
    COMPLEX = "complex"      # Major changes, significant effort required
    CRITICAL = "critical"    # Complete rewrite needed


@dataclass
class MigrationStep:
    """Individual migration step"""
    step_id: str
    title: str
    description: str
    complexity: MigrationComplexity
    estimated_time_minutes: int
    code_example: Optional[str] = None
    documentation_url: Optional[str] = None
    automated: bool = False
    breaking_change: bool = False
    required: bool = True


@dataclass
class MigrationPath:
    """Complete migration path between versions"""
    from_version: APIVersion
    to_version: APIVersion
    complexity: MigrationComplexity
    estimated_total_time_hours: float
    steps: List[MigrationStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    post_migration_tests: List[str] = field(default_factory=list)
    rollback_instructions: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MigrationRegistry:
    """Registry for managing migration paths"""
    
    def __init__(self):
        self.migration_paths: Dict[Tuple[APIVersion, APIVersion], MigrationPath] = {}
        self._initialize_default_migrations()
    
    def _initialize_default_migrations(self):
        """Initialize default migration paths"""
        
        # v1.0 to v1.1 migration
        self.register_migration_path(MigrationPath(
            from_version=APIVersion.V1_0,
            to_version=APIVersion.V1_1,
            complexity=MigrationComplexity.SIMPLE,
            estimated_total_time_hours=2.0,
            steps=[
                MigrationStep(
                    step_id="auth_fields",
                    title="Update Authentication Fields",
                    description="Replace deprecated field names in authentication requests",
                    complexity=MigrationComplexity.TRIVIAL,
                    estimated_time_minutes=15,
                    automated=True,
                    code_example='''
# Before (v1.0)
{
    "user_email": "researcher@university.edu",
    "user_password": "password123"
}

# After (v1.1)
{
    "email": "researcher@university.edu",
    "password": "password123"
}
''',
                    documentation_url="https://docs.prsm.org/migration/v1.0-to-v1.1/auth-fields"
                ),
                MigrationStep(
                    step_id="balance_fields",
                    title="Update FTNS Balance Field Names",
                    description="Replace legacy balance field names with new standardized names",
                    complexity=MigrationComplexity.TRIVIAL,
                    estimated_time_minutes=10,
                    automated=True,
                    code_example='''
# Before (v1.0)
{
    "token_balance": 1000.0,
    "locked_tokens": 100.0
}

# After (v1.1)
{
    "available_balance": 1000.0,
    "locked_balance": 100.0
}
''',
                    documentation_url="https://docs.prsm.org/migration/v1.0-to-v1.1/balance-fields"
                ),
                MigrationStep(
                    step_id="session_fields",
                    title="Update Session Management Fields",
                    description="Replace session field names for consistency",
                    complexity=MigrationComplexity.TRIVIAL,
                    estimated_time_minutes=10,
                    automated=True,
                    code_example='''
# Before (v1.0)
{
    "session_budget": 500.0,
    "session_spent": 125.0
}

# After (v1.1)
{
    "ftns_budget": 500.0,
    "ftns_spent": 125.0
}
''',
                    documentation_url="https://docs.prsm.org/migration/v1.0-to-v1.1/session-fields"
                ),
                MigrationStep(
                    step_id="enhanced_errors",
                    title="Handle Enhanced Error Responses",
                    description="Update error handling for improved error response format",
                    complexity=MigrationComplexity.SIMPLE,
                    estimated_time_minutes=30,
                    breaking_change=True,
                    code_example='''
# v1.1 enhanced error responses include more context
{
    "success": false,
    "error_code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient FTNS balance for transaction",
    "details": {
        "required_amount": 100.0,
        "available_balance": 50.0,
        "suggested_action": "Add more FTNS tokens to your account"
    },
    "timestamp": "2024-01-15T10:00:00Z",
    "request_id": "req_123456789"
}
''',
                    documentation_url="https://docs.prsm.org/migration/v1.0-to-v1.1/error-handling"
                ),
                MigrationStep(
                    step_id="update_sdks",
                    title="Update SDK Dependencies",
                    description="Update PRSM SDK to version compatible with v1.1",
                    complexity=MigrationComplexity.SIMPLE,
                    estimated_time_minutes=15,
                    code_example='''
# Python
pip install --upgrade prsm-sdk>=1.1.0

# JavaScript
npm install @prsm/js-sdk@^1.1.0

# Update client initialization
client = PRSMClient(api_version="1.1")
''',
                    documentation_url="https://docs.prsm.org/migration/v1.0-to-v1.1/sdk-update"
                )
            ],
            prerequisites=[
                "Backup current integration code",
                "Review API changelog for v1.1",
                "Test in staging environment first"
            ],
            post_migration_tests=[
                "Verify authentication flow works",
                "Test FTNS balance retrieval",
                "Confirm session management functionality",
                "Run end-to-end integration tests"
            ],
            rollback_instructions="Set API-Version header to '1.0' to rollback to previous version"
        ))
        
        # v1.1 to v2.0 migration (future)
        self.register_migration_path(MigrationPath(
            from_version=APIVersion.V1_1,
            to_version=APIVersion.V2_0,
            complexity=MigrationComplexity.COMPLEX,
            estimated_total_time_hours=8.0,
            steps=[
                MigrationStep(
                    step_id="auth_restructure",
                    title="Restructured Authentication Flow",
                    description="Migrate to new OAuth2-based authentication system",
                    complexity=MigrationComplexity.COMPLEX,
                    estimated_time_minutes=180,
                    breaking_change=True,
                    required=True,
                    documentation_url="https://docs.prsm.org/migration/v1.1-to-v2.0/auth-restructure"
                ),
                MigrationStep(
                    step_id="marketplace_api",
                    title="New Marketplace API Endpoints",
                    description="Update to new marketplace API structure with enhanced features",
                    complexity=MigrationComplexity.MODERATE,
                    estimated_time_minutes=120,
                    breaking_change=True,
                    documentation_url="https://docs.prsm.org/migration/v1.1-to-v2.0/marketplace-api"
                ),
                MigrationStep(
                    step_id="websocket_format",
                    title="Updated WebSocket Message Format",
                    description="Migrate to new WebSocket message structure",
                    complexity=MigrationComplexity.MODERATE,
                    estimated_time_minutes=90,
                    breaking_change=True,
                    documentation_url="https://docs.prsm.org/migration/v1.1-to-v2.0/websocket-format"
                )
            ],
            prerequisites=[
                "Complete comprehensive testing of current integration",
                "Plan for extended downtime during migration",
                "Review v2.0 API documentation thoroughly",
                "Prepare rollback plan"
            ]
        ))
    
    def register_migration_path(self, migration_path: MigrationPath):
        """Register a migration path"""
        key = (migration_path.from_version, migration_path.to_version)
        self.migration_paths[key] = migration_path
        logger.info(f"Registered migration path from {migration_path.from_version.value} to {migration_path.to_version.value}")
    
    def get_migration_path(self, from_version: APIVersion, to_version: APIVersion) -> Optional[MigrationPath]:
        """Get migration path between versions"""
        return self.migration_paths.get((from_version, to_version))
    
    def get_available_migrations(self, from_version: APIVersion) -> List[MigrationPath]:
        """Get all available migration paths from a version"""
        return [
            path for key, path in self.migration_paths.items()
            if key[0] == from_version
        ]


class DeprecationNotificationSystem:
    """System for managing deprecation notifications"""
    
    def __init__(self):
        self.notification_history: Dict[str, List[Dict[str, Any]]] = {}
        self.notification_rules: Dict[APIVersion, Dict[str, Any]] = {}
        self._initialize_notification_rules()
    
    def _initialize_notification_rules(self):
        """Initialize deprecation notification rules"""
        
        # Example: v1.0 deprecation rules
        self.notification_rules[APIVersion.V1_0] = {
            "warning_threshold_days": 90,  # Start showing warnings 90 days before sunset
            "notification_frequency": "weekly",  # How often to log notifications
            "escalation_days": [60, 30, 14, 7, 1],  # Days before sunset to escalate warnings
            "channels": ["response_headers", "logs", "email"]  # Where to send notifications
        }
    
    def should_notify(self, version: APIVersion, user_id: Optional[str] = None) -> bool:
        """Check if deprecation notification should be sent"""
        
        version_info = version_registry.get_version_info(version)
        if not version_info or not version_info.sunset_date:
            return False
        
        days_until_sunset = (version_info.sunset_date - datetime.now(timezone.utc)).days
        rules = self.notification_rules.get(version, {})
        warning_threshold = rules.get("warning_threshold_days", 90)
        
        if days_until_sunset <= warning_threshold:
            # Check if we should notify based on frequency
            return self._should_notify_by_frequency(version, user_id, rules)
        
        return False
    
    def _should_notify_by_frequency(self, version: APIVersion, user_id: Optional[str], rules: Dict[str, Any]) -> bool:
        """Check if notification should be sent based on frequency rules"""
        
        frequency = rules.get("notification_frequency", "weekly")
        notification_key = f"{version.value}:{user_id or 'anonymous'}"
        
        history = self.notification_history.get(notification_key, [])
        if not history:
            return True
        
        last_notification = max(history, key=lambda x: x["timestamp"])
        last_time = datetime.fromisoformat(last_notification["timestamp"])
        
        if frequency == "daily":
            return (datetime.now(timezone.utc) - last_time).days >= 1
        elif frequency == "weekly":
            return (datetime.now(timezone.utc) - last_time).days >= 7
        elif frequency == "monthly":
            return (datetime.now(timezone.utc) - last_time).days >= 30
        
        return False
    
    def record_notification(self, version: APIVersion, user_id: Optional[str], notification_type: str):
        """Record that a notification was sent"""
        
        notification_key = f"{version.value}:{user_id or 'anonymous'}"
        
        if notification_key not in self.notification_history:
            self.notification_history[notification_key] = []
        
        self.notification_history[notification_key].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "version": version.value
        })
    
    def get_deprecation_message(self, version: APIVersion) -> str:
        """Get appropriate deprecation message for version"""
        
        version_info = version_registry.get_version_info(version)
        if not version_info:
            return ""
        
        if version_info.sunset_date:
            days_until_sunset = (version_info.sunset_date - datetime.now(timezone.utc)).days
            
            if days_until_sunset <= 0:
                return f"API version {version.value} has been discontinued. Please upgrade immediately."
            elif days_until_sunset <= 7:
                return f"⚠️ URGENT: API version {version.value} will be discontinued in {days_until_sunset} days. Upgrade required."
            elif days_until_sunset <= 30:
                return f"⚠️ WARNING: API version {version.value} will be discontinued in {days_until_sunset} days. Please plan your upgrade."
            else:
                return f"API version {version.value} is deprecated and will be discontinued on {version_info.sunset_date.strftime('%B %d, %Y')}."
        
        return f"API version {version.value} is deprecated. Please consider upgrading to the latest version."


class MigrationAssistant:
    """Assistant for helping with API migrations"""
    
    def __init__(self, migration_registry: MigrationRegistry):
        self.registry = migration_registry
    
    def generate_migration_checklist(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Generate a comprehensive migration checklist"""
        
        migration_path = self.registry.get_migration_path(from_version, to_version)
        if not migration_path:
            return {"error": "No migration path found"}
        
        checklist = {
            "migration_overview": {
                "from_version": from_version.value,
                "to_version": to_version.value,
                "complexity": migration_path.complexity.value,
                "estimated_time_hours": migration_path.estimated_total_time_hours,
                "total_steps": len(migration_path.steps)
            },
            "prerequisites": [
                {"task": prereq, "completed": False} 
                for prereq in migration_path.prerequisites
            ],
            "migration_steps": [
                {
                    "step_id": step.step_id,
                    "title": step.title,
                    "description": step.description,
                    "complexity": step.complexity.value,
                    "estimated_minutes": step.estimated_time_minutes,
                    "automated": step.automated,
                    "breaking_change": step.breaking_change,
                    "required": step.required,
                    "completed": False,
                    "documentation_url": step.documentation_url,
                    "code_example": step.code_example
                }
                for step in migration_path.steps
            ],
            "post_migration_tests": [
                {"test": test, "completed": False}
                for test in migration_path.post_migration_tests
            ],
            "rollback_plan": migration_path.rollback_instructions
        }
        
        return checklist
    
    def get_breaking_changes_summary(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Get summary of breaking changes"""
        
        migration_path = self.registry.get_migration_path(from_version, to_version)
        if not migration_path:
            return {"error": "No migration path found"}
        
        breaking_changes = [
            {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "impact": "high" if step.complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.CRITICAL] else "medium",
                "required": step.required,
                "documentation_url": step.documentation_url
            }
            for step in migration_path.steps
            if step.breaking_change
        ]
        
        return {
            "total_breaking_changes": len(breaking_changes),
            "breaking_changes": breaking_changes,
            "migration_complexity": migration_path.complexity.value,
            "recommended_approach": self._get_recommended_approach(migration_path.complexity)
        }
    
    def _get_recommended_approach(self, complexity: MigrationComplexity) -> str:
        """Get recommended migration approach based on complexity"""
        
        approaches = {
            MigrationComplexity.TRIVIAL: "Automated migration - minimal testing required",
            MigrationComplexity.SIMPLE: "Gradual migration - test in staging first",
            MigrationComplexity.MODERATE: "Phased migration - plan for extended testing period",
            MigrationComplexity.COMPLEX: "Careful migration - extensive testing and rollback plan required",
            MigrationComplexity.CRITICAL: "Complete rewrite - consider it a new integration project"
        }
        
        return approaches.get(complexity, "Custom migration approach required")
    
    def validate_migration_readiness(self, from_version: APIVersion, to_version: APIVersion, 
                                   current_implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if current implementation is ready for migration"""
        
        migration_path = self.registry.get_migration_path(from_version, to_version)
        if not migration_path:
            return {"error": "No migration path found"}
        
        readiness_checks = []
        
        # Check each migration step
        for step in migration_path.steps:
            check_result = self._validate_step_readiness(step, current_implementation)
            readiness_checks.append(check_result)
        
        total_issues = sum(len(check["issues"]) for check in readiness_checks)
        overall_readiness = "ready" if total_issues == 0 else "needs_attention"
        
        return {
            "overall_readiness": overall_readiness,
            "total_issues": total_issues,
            "step_checks": readiness_checks
        }
    
    def _validate_step_readiness(self, step: MigrationStep, implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate readiness for a specific migration step"""
        
        issues = []
        warnings = []
        
        # Basic validation logic (would be more sophisticated in practice)
        if step.breaking_change and not implementation.get("has_error_handling", False):
            issues.append("Missing error handling for breaking changes")
        
        if step.complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.CRITICAL]:
            if not implementation.get("has_comprehensive_tests", False):
                warnings.append("Comprehensive testing recommended for complex migration")
        
        return {
            "step_id": step.step_id,
            "step_title": step.title,
            "ready": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


# Global instances
migration_registry = MigrationRegistry()
deprecation_system = DeprecationNotificationSystem()
migration_assistant = MigrationAssistant(migration_registry)


def create_migration_endpoints(app: FastAPI):
    """Create migration-related API endpoints"""
    
    @app.get("/api/migration/paths", include_in_schema=False)
    async def get_migration_paths(from_version: Optional[str] = None):
        """Get available migration paths"""
        
        if from_version:
            try:
                from_ver = APIVersion.from_string(from_version)
                paths = migration_registry.get_available_migrations(from_ver)
                return {
                    "from_version": from_version,
                    "available_migrations": [
                        {
                            "to_version": path.to_version.value,
                            "complexity": path.complexity.value,
                            "estimated_hours": path.estimated_total_time_hours,
                            "steps_count": len(path.steps)
                        }
                        for path in paths
                    ]
                }
            except ValueError:
                return {"error": f"Invalid version: {from_version}"}
        
        # Return all migration paths
        all_paths = []
        for (from_ver, to_ver), path in migration_registry.migration_paths.items():
            all_paths.append({
                "from_version": from_ver.value,
                "to_version": to_ver.value,
                "complexity": path.complexity.value,
                "estimated_hours": path.estimated_total_time_hours
            })
        
        return {"migration_paths": all_paths}
    
    @app.get("/api/migration/checklist", include_in_schema=False)
    async def get_migration_checklist(from_version: str, to_version: str):
        """Get migration checklist between versions"""
        
        try:
            from_ver = APIVersion.from_string(from_version)
            to_ver = APIVersion.from_string(to_version)
            
            checklist = migration_assistant.generate_migration_checklist(from_ver, to_ver)
            return checklist
            
        except ValueError as e:
            return {"error": str(e)}
    
    @app.get("/api/migration/breaking-changes", include_in_schema=False)
    async def get_breaking_changes(from_version: str, to_version: str):
        """Get breaking changes summary"""
        
        try:
            from_ver = APIVersion.from_string(from_version)
            to_ver = APIVersion.from_string(to_version)
            
            summary = migration_assistant.get_breaking_changes_summary(from_ver, to_ver)
            return summary
            
        except ValueError as e:
            return {"error": str(e)}
    
    @app.get("/migration/guide/{from_version}/to/{to_version}", 
             response_class=HTMLResponse, include_in_schema=False)
    async def get_migration_guide_page(from_version: str, to_version: str):
        """Get HTML migration guide page"""
        
        try:
            from_ver = APIVersion.from_string(from_version)
            to_ver = APIVersion.from_string(to_version)
            
            checklist = migration_assistant.generate_migration_checklist(from_ver, to_ver)
            
            # Generate HTML page
            html_content = generate_migration_guide_html(checklist)
            return HTMLResponse(content=html_content)
            
        except ValueError:
            return HTMLResponse(
                content="<h1>Migration Guide Not Found</h1><p>Invalid version numbers.</p>",
                status_code=404
            )


def generate_migration_guide_html(checklist: Dict[str, Any]) -> str:
    """Generate HTML migration guide page"""
    
    overview = checklist.get("migration_overview", {})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PRSM API Migration Guide - v{overview.get('from_version')} to v{overview.get('to_version')}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .overview {{ background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 20px 0; }}
            .step {{ border: 1px solid #bdc3c7; margin: 10px 0; border-radius: 6px; }}
            .step-header {{ background: #34495e; color: white; padding: 15px; cursor: pointer; }}
            .step-content {{ padding: 20px; display: none; }}
            .step.completed .step-header {{ background: #27ae60; }}
            .complexity {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .complexity.trivial {{ background: #d5f4e6; color: #27ae60; }}
            .complexity.simple {{ background: #ffeaa7; color: #f39c12; }}
            .complexity.moderate {{ background: #fab1a0; color: #e17055; }}
            .complexity.complex {{ background: #fd79a8; color: #e84393; }}
            .code-example {{ background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 4px; overflow-x: auto; margin: 10px 0; }}
            .checkbox {{ margin-right: 10px; }}
            .breaking-change {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 PRSM API Migration Guide</h1>
            <h2>v{overview.get('from_version')} → v{overview.get('to_version')}</h2>
            
            <div class="overview">
                <h3>Migration Overview</h3>
                <p><strong>Complexity:</strong> <span class="complexity {overview.get('complexity')}">{overview.get('complexity', '').title()}</span></p>
                <p><strong>Estimated Time:</strong> {overview.get('estimated_time_hours', 0)} hours</p>
                <p><strong>Total Steps:</strong> {overview.get('total_steps', 0)}</p>
            </div>
            
            <h2>📋 Prerequisites</h2>
            <ul>
    """
    
    for prereq in checklist.get("prerequisites", []):
        html += f'<li><input type="checkbox" class="checkbox">{prereq["task"]}</li>'
    
    html += """
            </ul>
            
            <h2>🔄 Migration Steps</h2>
    """
    
    for i, step in enumerate(checklist.get("migration_steps", []), 1):
        breaking_badge = ' <span class="breaking-change">[BREAKING CHANGE]</span>' if step.get("breaking_change") else ''
        automated_badge = ' <span style="color: #27ae60;">[AUTOMATED]</span>' if step.get("automated") else ''
        
        html += f"""
            <div class="step" id="step-{step['step_id']}">
                <div class="step-header" onclick="toggleStep('{step['step_id']}')">
                    <input type="checkbox" class="checkbox" onclick="event.stopPropagation(); markCompleted('{step['step_id']}')">
                    Step {i}: {step['title']}
                    <span class="complexity {step['complexity']}">{step['complexity'].title()}</span>
                    {breaking_badge}{automated_badge}
                    <span style="float: right;">~{step['estimated_minutes']} min</span>
                </div>
                <div class="step-content" id="content-{step['step_id']}">
                    <p>{step['description']}</p>
        """
        
        if step.get("code_example"):
            html += f'<div class="code-example"><pre><code>{step["code_example"]}</code></pre></div>'
        
        if step.get("documentation_url"):
            html += f'<p><a href="{step["documentation_url"]}" target="_blank">📖 View detailed documentation</a></p>'
        
        html += """
                </div>
            </div>
        """
    
    html += """
            <h2>🧪 Post-Migration Tests</h2>
            <ul>
    """
    
    for test in checklist.get("post_migration_tests", []):
        html += f'<li><input type="checkbox" class="checkbox">{test["test"]}</li>'
    
    html += f"""
            </ul>
            
            <h2>🔄 Rollback Plan</h2>
            <div class="overview">
                <p>{checklist.get('rollback_plan', 'No rollback plan specified.')}</p>
            </div>
        </div>
        
        <script>
            function toggleStep(stepId) {{
                const content = document.getElementById('content-' + stepId);
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            }}
            
            function markCompleted(stepId) {{
                const step = document.getElementById('step-' + stepId);
                step.classList.toggle('completed');
            }}
            
            // Initialize all steps as collapsed
            document.addEventListener('DOMContentLoaded', function() {{
                const contents = document.querySelectorAll('.step-content');
                contents.forEach(content => content.style.display = 'none');
            }});
        </script>
    </body>
    </html>
    """
    
    return html