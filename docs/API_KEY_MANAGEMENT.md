# API Key Management & External Service Authentication

## Overview

PRSM implements enterprise-grade API key management and external service authentication using encrypted credential storage. This system eliminates security vulnerabilities from storing API keys in environment variables and provides centralized credential management.

## Security Features

### 🔐 Encrypted Credential Storage

- **AES Encryption**: All credentials stored using Fernet (AES-128) encryption
- **Per-User Isolation**: User-specific and system-level credential separation
- **Automatic Rotation**: Support for credential rotation and expiry handling
- **Audit Logging**: Comprehensive logging of all credential access and operations

### 🛡️ Security Improvements

**Before (Insecure)**:
```python
# VULNERABLE: Direct environment variable access
openai_key = os.getenv('OPENAI_API_KEY')
```

**After (Secure)**:
```python
# SECURE: Encrypted credential manager
openai_client = await get_secure_api_client(SecureClientType.OPENAI, user_id)
```

## Supported Platforms

### AI Model Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo models
- **Anthropic**: Claude-3 models  
- **Hugging Face**: Community models and inference API

### Platform Integrations
- **GitHub**: Repository integration with OAuth
- **Pinecone**: Vector database service
- **Weaviate**: Vector database service
- **Ollama**: Local LLM runtime

## Quick Start

### 1. Initialize Secure Configuration

```bash
# Initialize the secure credential system
python scripts/manage_credentials.py initialize

# Migrate existing environment variables
python scripts/manage_credentials.py migrate
```

### 2. Register API Credentials

```bash
# Interactive registration for OpenAI
python scripts/manage_credentials.py register -p openai -i

# Direct registration
python scripts/manage_credentials.py register -p anthropic -k "your-api-key"

# GitHub OAuth token
python scripts/manage_credentials.py register -p github -t "your-github-token"
```

### 3. Validate Credentials

```bash
# Validate all platforms
python scripts/manage_credentials.py validate

# Validate specific platform
python scripts/manage_credentials.py validate -p openai
```

## API Usage

### REST API Endpoints

#### Register Credentials
```http
POST /api/v1/credentials/register
Content-Type: application/json
Authorization: Bearer <jwt-token>

{
    "platform": "openai",
    "credentials": {
        "api_key": "your-secure-api-key"
    },
    "user_specific": false,
    "expires_at": "2024-12-31T23:59:59Z"
}
```

#### Validate Credentials
```http
POST /api/v1/credentials/validate
Content-Type: application/json
Authorization: Bearer <jwt-token>

{
    "platform": "openai",
    "user_specific": false
}
```

#### Get Credential Status
```http
GET /api/v1/credentials/status
Authorization: Bearer <jwt-token>
```

#### System Status (Admin Only)
```http
GET /api/v1/credentials/system/status
Authorization: Bearer <jwt-token>
```

### Python SDK Usage

#### Get Secure API Client
```python
from prsm.integrations.security.secure_api_client_factory import get_secure_api_client, SecureClientType

# Get OpenAI client for user
openai_client = await get_secure_api_client(SecureClientType.OPENAI, user_id)
if openai_client:
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

# Get system-level client
anthropic_client = await get_system_api_client(SecureClientType.ANTHROPIC)
```

#### Register Credentials Programmatically
```python
from prsm.integrations.security.secure_config_manager import secure_config_manager

success = await secure_config_manager.register_api_credentials(
    platform="openai",
    credentials={"api_key": "your-key"},
    user_id="user123"  # or None for system
)
```

## CLI Commands

### Core Commands

```bash
# Initialize secure configuration
python scripts/manage_credentials.py initialize

# List supported platforms
python scripts/manage_credentials.py list-platforms

# Get system status
python scripts/manage_credentials.py status

# Get status as JSON
python scripts/manage_credentials.py status --json-output
```

### Credential Management

```bash
# Register credentials interactively
python scripts/manage_credentials.py register -p openai -i

# Register with direct input
python scripts/manage_credentials.py register -p anthropic -k "your-key"

# Register GitHub token
python scripts/manage_credentials.py register -p github -t "ghp_token"

# Register Pinecone with environment
python scripts/manage_credentials.py register -p pinecone -k "key" -e "us-west1-gcp"

# Register Weaviate with custom URL
python scripts/manage_credentials.py register -p weaviate -u "http://localhost:8080"
```

### Validation & Status

```bash
# Validate all credentials
python scripts/manage_credentials.py validate

# Validate specific platform
python scripts/manage_credentials.py validate -p openai

# Validate for specific user
python scripts/manage_credentials.py validate --user-id user123
```

## Platform-Specific Configuration

### OpenAI
```bash
python scripts/manage_credentials.py register -p openai -i
# Requires: API key from https://platform.openai.com/api-keys
```

### Anthropic  
```bash
python scripts/manage_credentials.py register -p anthropic -i
# Requires: API key from https://console.anthropic.com/
```

### Hugging Face
```bash
python scripts/manage_credentials.py register -p huggingface -i
# Requires: Access token from https://huggingface.co/settings/tokens
```

### GitHub
```bash
python scripts/manage_credentials.py register -p github -t "ghp_your_token"
# Requires: Personal access token or OAuth app token
```

### Pinecone
```bash
python scripts/manage_credentials.py register -p pinecone -k "your-key" -e "us-west1-gcp"
# Requires: API key and environment from Pinecone console
```

### Weaviate
```bash
python scripts/manage_credentials.py register -p weaviate -u "http://localhost:8080"
# Optional: API key for cloud instances
```

### Ollama
```bash
python scripts/manage_credentials.py register -p ollama -u "http://localhost:11434"
# Typically no API key required for local instances
```

## Migration from Environment Variables

### Automatic Migration

The system automatically migrates environment variables to encrypted storage during initialization:

```bash
# Migrate all detected environment variables
python scripts/manage_credentials.py migrate
```

**Detected Environment Variables**:
- `OPENAI_API_KEY` → OpenAI credentials
- `ANTHROPIC_API_KEY` → Anthropic credentials
- `HUGGINGFACE_API_KEY` → Hugging Face credentials
- `GITHUB_ACCESS_TOKEN` → GitHub credentials
- `PINECONE_API_KEY` → Pinecone credentials

### Manual Migration Steps

1. **Initialize secure system**:
   ```bash
   python scripts/manage_credentials.py initialize
   ```

2. **Register credentials manually**:
   ```bash
   python scripts/manage_credentials.py register -p openai -k "$OPENAI_API_KEY"
   ```

3. **Validate migration**:
   ```bash
   python scripts/manage_credentials.py validate
   ```

4. **Remove environment variables** from `.env` files

## Security Best Practices

### For Developers

1. **Never use environment variables for production credentials**
2. **Use the secure client factory for all external API access**
3. **Validate credentials before using them**
4. **Log credential access for audit trails**
5. **Rotate credentials regularly**

### For System Administrators

1. **Initialize secure configuration on deployment**
2. **Monitor credential usage and expiry**
3. **Review audit logs regularly**
4. **Use system-level credentials for background services**
5. **Implement credential rotation policies**

### For DevOps Teams

1. **Use credential management API in CI/CD pipelines**
2. **Separate development and production credentials**
3. **Monitor credential access patterns**
4. **Implement automated credential health checks**

## Monitoring & Auditing

### Credential Status Monitoring

```python
# Get comprehensive status
status = await secure_config_manager.get_secure_configuration_status()

# Check specific platform
is_valid = await secure_client_factory.validate_client_credentials(
    SecureClientType.OPENAI, user_id
)
```

### Audit Logging

All credential operations are logged with:
- **User ID**: Who accessed credentials
- **Platform**: Which service was accessed
- **Operation**: What operation was performed
- **Timestamp**: When the operation occurred
- **Success/Failure**: Whether the operation succeeded

### Health Checks

```bash
# Check system health
curl -H "Authorization: Bearer $JWT_TOKEN" \
     http://localhost:8000/api/v1/credentials/system/status

# Validate platform credentials
curl -X POST -H "Authorization: Bearer $JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"platform": "openai"}' \
     http://localhost:8000/api/v1/credentials/validate
```

## Troubleshooting

### Common Issues

1. **Credentials not found**
   ```bash
   # Check if credentials are registered
   python scripts/manage_credentials.py status
   
   # Register missing credentials
   python scripts/manage_credentials.py register -p openai -i
   ```

2. **Migration not working**
   ```bash
   # Check environment variables
   env | grep -E "(OPENAI|ANTHROPIC|HUGGINGFACE)_API_KEY"
   
   # Initialize system first
   python scripts/manage_credentials.py initialize
   ```

3. **Permission denied errors**
   - Ensure user has proper authentication
   - Check JWT token validity
   - Verify user permissions for system operations

4. **Validation failures**
   ```bash
   # Test credentials manually
   python scripts/manage_credentials.py validate -p openai
   
   # Re-register credentials if invalid
   python scripts/manage_credentials.py register -p openai -i
   ```

### Debug Steps

1. **Check system status**:
   ```bash
   python scripts/manage_credentials.py status
   ```

2. **Validate configuration**:
   ```bash
   python scripts/manage_credentials.py validate
   ```

3. **Review logs**:
   - Check PRSM server logs for credential access attempts
   - Look for audit log entries in security logs

4. **Test API endpoints**:
   ```bash
   curl -H "Authorization: Bearer $JWT_TOKEN" \
        http://localhost:8000/api/v1/credentials/status
   ```

## Configuration Files

### Secure Environment Template

A secure `.env` template is generated at `config/secure.env.template`:

```bash
# PRSM Secure Configuration Template
SECRET_KEY=GENERATE_SECURE_RANDOM_STRING_64_CHARS_MINIMUM
DATABASE_URL=postgresql://username:password@localhost:5432/prsm
REDIS_URL=redis://localhost:6379/0

# API keys managed via credential management system
# Use: python scripts/manage_credentials.py register -p <platform> -i
```

### Production Deployment

For production deployment:

1. **Generate secure secrets**
2. **Initialize credential management**
3. **Register all required API credentials**
4. **Remove any hardcoded credentials**
5. **Enable audit logging**

## Integration Examples

### Custom Model Executor

```python
from prsm.integrations.security.secure_api_client_factory import get_secure_api_client, SecureClientType

class SecureModelExecutor:
    async def execute_openai_model(self, user_id: str, prompt: str):
        # Get secure OpenAI client
        client = await get_secure_api_client(SecureClientType.OPENAI, user_id)
        if not client:
            raise ValueError("OpenAI credentials not available")
        
        # Use client securely
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

### Background Service

```python
from prsm.integrations.security.secure_api_client_factory import get_system_api_client, SecureClientType

class BackgroundModelService:
    async def process_queue(self):
        # Use system credentials for background processing
        anthropic_client = await get_system_api_client(SecureClientType.ANTHROPIC)
        if anthropic_client:
            # Process tasks with system-level credentials
            pass
```

This comprehensive API key management system ensures that all external service integrations in PRSM are secure, audited, and properly managed for production deployment.