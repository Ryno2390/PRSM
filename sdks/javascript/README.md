# PRSM JavaScript/TypeScript SDK

Official JavaScript and TypeScript client for the Protocol for Recursive Scientific Modeling (PRSM).

[![npm version](https://badge.fury.io/js/%40prsm%2Fsdk.svg)](https://badge.fury.io/js/%40prsm%2Fsdk)
[![TypeScript](https://img.shields.io/badge/TypeScript-ready-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

## Installation

```bash
npm install @prsm/sdk
# or
yarn add @prsm/sdk
# or
pnpm add @prsm/sdk
```

## Quick Start

### TypeScript

```typescript
import { PRSMClient } from '@prsm/sdk';

async function main() {
  // Initialize client
  const client = new PRSMClient({ apiKey: 'your_api_key_here' });
  
  // Simple AI query
  const response = await client.query('Explain quantum computing in simple terms');
  console.log(response.content);
  
  // Check token balance
  const balance = await client.ftns.getBalance();
  console.log(`FTNS Balance: ${balance.availableBalance}`);
  
  // Search for models
  const models = await client.marketplace.searchModels({ query: 'gpt' });
  console.log(`Found ${models.length} models`);
  
  // Clean up
  await client.close();
}

main().catch(console.error);
```

### JavaScript (ES6+)

```javascript
const { PRSMClient } = require('@prsm/sdk');

async function main() {
  const client = new PRSMClient({ apiKey: 'your_api_key_here' });
  
  const response = await client.query('Hello, PRSM!');
  console.log(response.content);
  
  await client.close();
}

main().catch(console.error);
```

## Features

- 🤖 **AI Query Interface** - Simple access to PRSM's distributed AI network
- 💰 **FTNS Token Management** - Built-in token balance and cost tracking
- 🔒 **Authentication** - Secure API key and JWT token handling
- 📊 **Model Marketplace** - Browse and use community-contributed models
- 🛡️ **Safety Integration** - Built-in safety monitoring and circuit breakers
- 📈 **Performance Monitoring** - Request tracking and optimization
- 🌐 **P2P Network Access** - Direct access to distributed computing resources
- 🔧 **Tool Integration** - MCP tool protocol support for enhanced capabilities
- 📝 **TypeScript Support** - Full type definitions and IntelliSense support

## Advanced Usage

### Streaming Responses

```typescript
import { PRSMClient } from '@prsm/sdk';

async function streamExample() {
  const client = new PRSMClient({ apiKey: 'your_api_key' });
  
  console.log('AI Response: ');
  for await (const chunk of client.stream('Write a short story about AI')) {
    process.stdout.write(chunk.content);
  }
  console.log(); // New line
  
  await client.close();
}
```

### Cost Estimation

```typescript
async function costExample() {
  const client = new PRSMClient({ apiKey: 'your_api_key' });
  
  // Estimate cost before running
  const cost = await client.estimateCost('Complex scientific query about protein folding');
  console.log(`Estimated cost: ${cost} FTNS`);
  
  // Check if we have enough balance
  const balance = await client.ftns.getBalance();
  if (balance.availableBalance >= cost) {
    const response = await client.query('Complex scientific query about protein folding');
    console.log(response.content);
  } else {
    console.log('Insufficient FTNS balance');
  }
  
  await client.close();
}
```

### Model Marketplace

```typescript
import { ModelProvider } from '@prsm/sdk';

async function marketplaceExample() {
  const client = new PRSMClient({ apiKey: 'your_api_key' });
  
  // Search for specific models
  const scienceModels = await client.marketplace.searchModels({
    query: 'scientific research',
    provider: ModelProvider.HUGGINGFACE,
    minPerformance: 0.8,
    maxCost: 0.001,
    limit: 10
  });
  
  // Use a specific model
  if (scienceModels.length > 0) {
    const model = scienceModels[0];
    const response = await client.query('Explain CRISPR gene editing', {
      modelId: model.id
    });
    console.log(`Response from ${model.name}: ${response.content}`);
  }
  
  await client.close();
}
```

### Tool Execution (MCP)

```typescript
async function toolsExample() {
  const client = new PRSMClient({ apiKey: 'your_api_key' });
  
  // List available tools
  const tools = await client.tools.listAvailable();
  console.log(`Available tools: ${tools.map(t => t.name).join(', ')}`);
  
  // Execute a tool
  const webSearchTool = tools.find(t => t.name === 'web_search');
  if (webSearchTool) {
    const result = await client.tools.execute({
      toolName: 'web_search',
      parameters: { query: 'latest AI research papers' }
    });
    console.log(`Search results: ${JSON.stringify(result.result, null, 2)}`);
  }
  
  await client.close();
}
```

### Error Handling

```typescript
import { 
  PRSMClient, 
  InsufficientFundsError, 
  SafetyViolationError,
  AuthenticationError 
} from '@prsm/sdk';

async function errorHandlingExample() {
  try {
    const client = new PRSMClient({ apiKey: 'invalid_key' });
    const response = await client.query('Hello world');
  } catch (error) {
    if (error instanceof AuthenticationError) {
      console.log('Invalid API key');
    } else if (error instanceof InsufficientFundsError) {
      console.log(`Not enough FTNS: ${error.message}`);
    } else if (error instanceof SafetyViolationError) {
      console.log(`Content safety violation: ${error.message}`);
    } else {
      console.error('Unexpected error:', error);
    }
  }
}
```

## Configuration

### Environment Variables

```bash
export PRSM_API_KEY="your_api_key_here"
export PRSM_BASE_URL="https://api.prsm.ai/v1"  # Optional
```

### Custom Configuration

```typescript
import { PRSMClient, PRSMClientConfig } from '@prsm/sdk';

const config: PRSMClientConfig = {
  apiKey: 'your_key',
  baseUrl: 'https://custom-api.prsm.ai/v1',
  websocketUrl: 'wss://custom-ws.prsm.ai/v1',
  timeout: 120000, // 2 minutes in milliseconds
  maxRetries: 5,
  headers: {
    'Custom-Header': 'value'
  }
};

const client = new PRSMClient(config);
```

## Browser Usage

The SDK works in both Node.js and modern browsers:

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import { PRSMClient } from 'https://unpkg.com/@prsm/sdk@latest/dist/index.esm.js';
    
    async function main() {
      const client = new PRSMClient({ apiKey: 'your_api_key' });
      const response = await client.query('Hello from the browser!');
      document.getElementById('response').textContent = response.content;
      await client.close();
    }
    
    main().catch(console.error);
  </script>
</head>
<body>
  <div id="response">Loading...</div>
</body>
</html>
```

## API Reference

### PRSMClient

The main client class for interacting with PRSM.

```typescript
class PRSMClient {
  constructor(config: PRSMClientConfig);
  
  query(prompt: string, options?: QueryRequest): Promise<PRSMResponse>;
  stream(prompt: string, options?: QueryRequest): AsyncIterable<StreamChunk>;
  estimateCost(prompt: string, modelId?: string): Promise<number>;
  getSafetyStatus(): Promise<SafetyStatus>;
  listAvailableModels(): Promise<ModelInfo[]>;
  healthCheck(): Promise<Record<string, any>>;
  close(): Promise<void>;
  
  // Managers
  readonly auth: AuthManager;
  readonly ftns: FTNSManager;
  readonly marketplace: ModelMarketplace;
  readonly tools: ToolExecutor;
}
```

### Type Definitions

All types are exported and fully documented:

```typescript
interface PRSMResponse {
  content: string;
  modelId: string;
  provider: ModelProvider;
  executionTime: number;
  tokenUsage: Record<string, number>;
  ftnsCost: number;
  reasoningTrace?: string[];
  safetyStatus: SafetyLevel;
  metadata: Record<string, any>;
  requestId: string;
  timestamp: string;
}

enum ModelProvider {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  HUGGINGFACE = 'huggingface',
  LOCAL = 'local',
  PRSM_DISTILLED = 'prsm_distilled'
}

enum SafetyLevel {
  NONE = 'none',
  LOW = 'low',
  MODERATE = 'moderate',
  HIGH = 'high',
  CRITICAL = 'critical',
  EMERGENCY = 'emergency'
}
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/PRSM-AI/PRSM.git
cd PRSM/sdks/javascript

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Run linting
npm run lint

# Generate documentation
npm run docs
```

### Building

```bash
# Build for production
npm run build

# Build and watch for changes
npm run build:watch
```

### Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## Examples

See the [examples](examples/) directory for more comprehensive examples:

- [Basic Usage](examples/basic-usage.js)
- [TypeScript Usage](examples/typescript-usage.ts)
- [Streaming Responses](examples/streaming.js)
- [Marketplace Integration](examples/marketplace.js)
- [Tool Execution](examples/tools.js)
- [React Integration](examples/react-example.jsx)

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Support

- 📚 **Documentation**: [docs.prsm.ai/javascript-sdk](https://docs.prsm.ai/javascript-sdk)
- 🐛 **Issues**: [GitHub Issues](https://github.com/PRSM-AI/PRSM/issues)
- 💬 **Community**: [Discord](https://discord.gg/prsm)
- 📧 **Email**: [dev@prsm.ai](mailto:dev@prsm.ai)