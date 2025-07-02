# PRSM Secure Setup Guide

## API Key Security

**⚠️ CRITICAL SECURITY NOTICE**: Never commit API keys to version control. The PRSM repository uses environment variables and secure configuration to protect your credentials.

## Quick Setup

1. **Copy the template file**:
   ```bash
   cp ai-concierge/API_Keys.example.txt ai-concierge/API_Keys.txt
   ```

2. **Add your API keys** to `ai-concierge/API_Keys.txt`:
   - Get Anthropic API key from: https://console.anthropic.com/
   - Get Google API key from: https://aistudio.google.com/app/apikey
   - Get OpenAI API key from: https://platform.openai.com/api-keys

3. **Environment variables** will be automatically configured from your API_Keys.txt file.

## Security Best Practices

### ✅ What PRSM Does for Security

- **Automatic .gitignore protection**: API_Keys.txt and .env files are never committed
- **Template files**: Only placeholder values are stored in version control
- **Environment isolation**: Development and production keys are separated
- **Secure defaults**: All configuration files use placeholder values by default

### 🔒 What You Should Do

1. **Keep API keys secure**:
   - Never share your API_Keys.txt file
   - Don't paste keys in chat messages or emails
   - Use different keys for development and production

2. **Monitor usage**:
   - Set spending limits on your API provider dashboards
   - Monitor usage regularly for unexpected activity
   - Rotate keys periodically

3. **Production deployment**:
   - Use environment variables in production
   - Never deploy with hardcoded keys
   - Use secrets management services (AWS Secrets Manager, etc.)

## File Structure

```
ai-concierge/
├── API_Keys.txt          # Your actual keys (never committed)
├── API_Keys.example.txt  # Template file (safe to commit)
├── .env                  # Auto-generated environment file
└── .env.example          # Template environment file
```

## Troubleshooting

### "API key not found" errors
1. Verify your API_Keys.txt file exists and contains valid keys
2. Check that keys don't have extra spaces or newlines
3. Restart the application after adding keys

### Keys not working
1. Verify keys are active in your provider dashboards
2. Check spending limits haven't been exceeded
3. Ensure keys have proper permissions for the APIs being used

## Emergency: Exposed Keys

If you accidentally expose API keys:

1. **Immediately revoke** the exposed keys in your provider dashboards
2. **Generate new keys** and update your local configuration
3. **Check git history** to ensure keys weren't committed:
   ```bash
   git log --all -p | grep -i "sk-"
   ```
4. **Contact your API providers** if you suspect unauthorized usage

## Support

For security questions or if you discover vulnerabilities, please create an issue in the PRSM repository with the "security" label.