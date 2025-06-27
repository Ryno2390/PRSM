# PRSM SDK Documentation

Comprehensive documentation for PRSM Software Development Kits (SDKs) across multiple programming languages.

## 🚀 Available SDKs

### Python SDK
- **Location**: [sdks/python/](../sdks/python/)
- **Features**: Full async/await support, type hints, enterprise features
- **Installation**: `pip install prsm-sdk`
- **Documentation**: [Python SDK Guide](../sdks/python/README.md)

### JavaScript/TypeScript SDK
- **Location**: [sdks/javascript/](../sdks/javascript/)
- **Features**: Browser + Node.js support, TypeScript definitions, React hooks
- **Installation**: `npm install @prsm/sdk`
- **Documentation**: [JavaScript SDK Guide](../sdks/javascript/README.md)

### Go SDK
- **Location**: [sdks/go/](../sdks/go/)
- **Features**: High-performance, concurrent operations, enterprise integration
- **Installation**: `go get github.com/Ryno2390/PRSM/sdks/go`
- **Documentation**: [Go SDK Guide](../sdks/go/README.md)

## 📚 Quick Start Guides

### Python Quick Start
```python
from prsm_sdk import Client

client = Client(api_key="your-api-key")
response = await client.query("Hello PRSM!")
print(response.text)
```

### JavaScript Quick Start
```javascript
import { Client } from '@prsm/sdk';

const client = new Client({ apiKey: 'your-api-key' });
const response = await client.query('Hello PRSM!');
console.log(response.text);
```

### Go Quick Start
```go
import "github.com/Ryno2390/PRSM/sdks/go/prsm"

client := prsm.NewClient("your-api-key")
response, err := client.Query("Hello PRSM!")
fmt.Println(response.Text)
```

## 🔧 Core Features

All SDKs support:

### Query Operations
- Text generation and completion
- Streaming responses
- Batch processing
- Cost estimation

### Marketplace Integration
- FTNS token operations
- Model marketplace access
- Usage tracking
- Payment processing

### Agent Framework
- Multi-agent coordination
- Tool execution
- Workflow orchestration
- State management

### Enterprise Features
- Authentication and authorization
- Audit logging
- Rate limiting
- Error handling and retries

## 📖 Advanced Documentation

### Integration Guides
- [API Reference](./API_REFERENCE.md)
- [Authentication Guide](./tutorials/02-foundation/configuration.md)
- [Error Handling Best Practices](./TROUBLESHOOTING_GUIDE.md)
- [Performance Optimization](./performance/PERFORMANCE_ASSESSMENT.md)

### Examples Repository
- [Python Examples](../sdks/python/examples/)
- [JavaScript Examples](../sdks/javascript/examples/)
- [Integration Examples](../docs/integration-guides/)

### Enterprise Documentation
- [Security Hardening](./SECURITY_HARDENING.md)
- [Production Operations](./PRODUCTION_OPERATIONS_MANUAL.md)
- [Monitoring and Observability](./performance/PERFORMANCE_INSTRUMENTATION_REPORT.md)

## 🤝 Community & Support

### Resources
- **GitHub Issues**: [Report Issues](https://github.com/Ryno2390/PRSM/issues)
- **Documentation**: [Complete Docs](../docs/)
- **API Reference**: [OpenAPI Spec](./api/)

### Contributing
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Development Setup](./tutorials/03-development/)
- [SDK Development Guide](./tutorials/04-distribution/)

## 📊 SDK Comparison

| Feature | Python | JavaScript | Go |
|---------|--------|------------|-----|
| Async Support | ✅ | ✅ | ✅ |
| Type Safety | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |
| Browser Support | ❌ | ✅ | ❌ |
| Performance | 🟡 | 🟡 | ✅ |
| Enterprise Features | ✅ | ✅ | ✅ |

## 🔄 Version Compatibility

| SDK Version | PRSM API | Python | Node.js | Go |
|-------------|----------|--------|---------|-----|
| 1.0.x | v1 | 3.8+ | 16+ | 1.19+ |
| 0.9.x | v1-beta | 3.7+ | 14+ | 1.18+ |

## 📞 Support

For SDK-specific support:
- **Email**: sdk-support@prsm.ai
- **Documentation Issues**: [GitHub Issues](https://github.com/Ryno2390/PRSM/issues)
- **Community**: [Discord](https://discord.gg/prsm)