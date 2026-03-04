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

## Node Bootstrap Configuration

### Bootstrap Behavior Overview

When a PRSM node starts, it connects to bootstrap peers to join the network. The bootstrap system follows a priority-ordered strategy:

1. **Primary nodes** (configured via `bootstrap_nodes` in node config or `--bootstrap` CLI flag) are tried first.
2. **Fallback nodes** (trusted PRSM community relays) are tried if all primary nodes are unreachable.
3. If all candidates fail, the node starts in **degraded local mode** — it remains functional for local operations but cannot discover remote peers until inbound connections arrive or bootstrap targets recover.

### Configuration Reference

| Setting | Default | Description |
|---|---|---|
| `bootstrap_nodes` | `["wss://bootstrap.prsm-network.com"]` | Primary bootstrap peers (tried first) |
| `bootstrap_fallback_enabled` | `true` | Enable/disable trusted fallback peers |
| `bootstrap_fallback_nodes` | `["wss://fallback1.prsm-network.com", "wss://fallback2.prsm-network.com"]` | Fallback peers (tried after all primaries fail) |
| `bootstrap_validate_addresses` | `true` | Reject malformed bootstrap addresses before connection attempts |
| `bootstrap_retry_attempts` | `2` | Number of connection attempts per bootstrap node |
| `bootstrap_connect_timeout` | `5.0` | Seconds to wait for each connection attempt |
| `bootstrap_backoff_base` | `1.0` | Base delay (seconds) for exponential backoff between retries |
| `bootstrap_backoff_max` | `8.0` | Maximum backoff delay cap (seconds) |

### Address Validation

When `bootstrap_validate_addresses` is enabled (default), the following addresses are rejected:

- Empty or whitespace-only strings
- URLs with non-WebSocket schemes (anything other than `ws://` or `wss://`)
- Addresses with non-numeric port numbers
- Addresses with ports outside the 1–65535 range

Valid address formats:
- `wss://bootstrap.example.com` (URL with scheme)
- `wss://bootstrap.example.com:9001` (URL with explicit port)
- `host:9001` (bare host:port)

### Failure Semantics

- **Primary failure, fallback success**: Node logs an info-level message indicating fallback activation. Node is fully operational.
- **All candidates fail**: Node starts in degraded local mode. The CLI preflight table shows a `WARN` status for bootstrap reachability. Peer discovery resumes when inbound peers connect or configured targets recover.
- **Malformed address rejection**: Addresses that fail validation are logged at WARN level with the rejection reason. They are never sent to the transport layer.

### Disabling Fallback Behavior

To disable the fallback bootstrap system entirely, set `bootstrap_fallback_enabled` to `false` in your node config (`~/.prsm/node_config.json`):

```json
{
  "bootstrap_fallback_enabled": false
}
```

## Support

For security questions or if you discover vulnerabilities, please create an issue in the PRSM repository with the "security" label.