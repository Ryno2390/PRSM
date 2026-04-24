# PRSM SDKs

Official Software Development Kits for the Protocol for Research, Storage, and Modeling (PRSM).

## Available SDKs

### 🐍 Python SDK (`python/`)
- **Package:** `prsm-python-sdk` — **v0.2.0 published ✅**
- **PyPI:** `pip install prsm-python-sdk`
- **Package page:** [pypi.org/project/prsm-python-sdk](https://pypi.org/project/prsm-python-sdk)
- **Documentation:** [Python SDK Docs](python/README.md)

### 🟨 JavaScript SDK (`javascript/`)
- **Package:** `prsm-sdk` — **v0.2.0 published ✅**
- **npm:** `npm install prsm-sdk`
- **Package page:** [npmjs.com/package/prsm-sdk](https://npmjs.com/package/prsm-sdk)
- **Documentation:** [JavaScript SDK Docs](javascript/README.md)

### 🐹 Go SDK (`go/`)
- **Module:** `github.com/prsm-network/PRSM/sdks/go` — **v0.2.0 published ✅**
- **Installation:** `go get github.com/prsm-network/PRSM/sdks/go@v0.2.0`
- **Package page:** [pkg.go.dev/github.com/prsm-network/PRSM/sdks/go](https://pkg.go.dev/github.com/prsm-network/PRSM/sdks/go)
- **Documentation:** [Go SDK Docs](go/README.md)

## Quick Start

### Python
```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your_api_key")
response = await client.query("Explain quantum computing")
print(response.content)
```

### JavaScript
```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({ apiKey: 'your_api_key' });
const response = await client.query('Explain quantum computing');
console.log(response.content);
```

### Go
```go
import "github.com/prsm-network/PRSM/sdks/go/client"

client := client.New("your_api_key")
response, err := client.Query("Explain quantum computing")
fmt.Println(response.Content)
```

## Features

All SDKs provide:

- 🤖 **AI Query Interface** - Simple access to PRSM's AI capabilities
- 💰 **FTNS Token Management** - Built-in token balance and cost tracking
- 🔒 **Authentication** - Secure API key and JWT token handling
- 📊 **Model Marketplace** - Access to PRSM's model ecosystem
- 🛡️ **Safety Integration** - Built-in safety monitoring and circuit breakers
- 📈 **Performance Monitoring** - Request tracking and optimization
- 🌐 **P2P Network** - Direct access to distributed computing resources
- 🔧 **Tool Integration** - MCP tool protocol support

## Development

Each SDK is independently developed and maintained with:

- Comprehensive test suites
- Type definitions (TypeScript/Python typing/Go interfaces)
- Auto-generated documentation from code
- CI/CD pipeline for automated testing and publishing
- Semantic versioning

## Contributing

See individual SDK directories for language-specific contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) file for details.