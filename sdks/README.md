# PRSM SDKs

Official Software Development Kits for the Protocol for Research, Storage, and Modeling (PRSM).

## Available SDKs

### 🐍 Python SDK (`prsm-network`)
- **Package:** `prsm-network` — published ✅
- **PyPI:** `pip install prsm-network`
- **Package page:** [pypi.org/project/prsm-network](https://pypi.org/project/prsm-network)
- **Documentation:** [SDK Developer Guide](../docs/SDK_DEVELOPER_GUIDE.md)

### 🟨 JavaScript SDK (`javascript/`)
- **Package:** `prsm-sdk` — published ✅
- **npm:** `npm install prsm-sdk`
- **Package page:** [npmjs.com/package/prsm-sdk](https://npmjs.com/package/prsm-sdk)
- **Documentation:** [JavaScript SDK Docs](javascript/README.md)

### 🐹 Go SDK (`go/`)
- **Module:** `github.com/prsm-network/PRSM/sdks/go` — **v0.37.0 published ✅**
- **Installation:** `go get github.com/prsm-network/PRSM/sdks/go@v0.37.0`
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

- 🤖 **Compute Pipeline Client** - Drive PRSM's Ring 1-10 quote/run/status pipeline
- 💰 **FTNS Token Management** - Built-in token balance and cost tracking
- 🔒 **Authentication** - Secure API key and JWT token handling
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