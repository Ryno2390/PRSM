# PRSM JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for **PRSM — a P2P infrastructure protocol for open-source collaboration**. PRSM aggregates consumer-node storage, compute, and data into a mesh network that any third-party LLM can reach through MCP tools. This SDK lets your JavaScript or TypeScript application drive the PRSM infrastructure layer directly.

[![npm version](https://badge.fury.io/js/%40prsm%2Fsdk.svg)](https://badge.fury.io/js/%40prsm%2Fsdk)
[![TypeScript](https://img.shields.io/badge/%3C%2F%3E-TypeScript-%230074c1.svg)](https://www.typescriptlang.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Ring 1-10 Compute Client** — submit quotes and queries through the full PRSM pipeline
- **TypeScript Support** — comprehensive type definitions for enhanced DX
- **Real-time WebSocket** — live job progress updates and streaming results
- **Error Handling** — robust error handling with automatic retry logic
- **Authentication** — JWT tokens, API keys, and secure session management
- **FTNS Integration** — balance checks, transfers, yield estimation
- **Storage / ContentStore** — upload content with royalty tracking, download by CID
- **MCP Tool Surface** — drive the same 16 tools that third-party LLMs use
- **Cross-platform** — Node.js, browsers, React Native

PRSM itself does not host models — reasoning happens in your LLM of choice. The SDK drives PRSM's compute dispatch, storage, and FTNS settlement layers.

## Installation

```bash
npm install @prsm/sdk
# or
yarn add @prsm/sdk
# or
pnpm add @prsm/sdk
```

## Quick Start

### Basic Usage

```typescript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  apiKey: 'your_api_key_here',
  baseUrl: 'https://api.prsm-network.com'
});

// Get a free cost quote
const quote = await client.compute.quote('EV adoption trends in NC', {
  shards: 5,
  tier: 't2',
});
console.log(`Estimated cost: ${quote.totalFtns} FTNS`);

// Execute the query through the Ring 1-10 pipeline
const result = await client.compute.run({
  query: 'EV adoption trends in NC',
  budget: quote.totalFtns * 1.1,
  privacy: 'standard',
});

console.log(result.content);
console.log(`FTNS spent: ${result.ftnsSpent}`);
```

## Core Features

> **Note:** The legacy NWTN research-session API, SEAL autonomous improvement API, hosted marketplace, and centralized REST governance were removed in v1.6.0. PRSM is now a P2P infrastructure protocol — reasoning happens in your third-party LLM via MCP, not inside PRSM. Governance is on-network stake-weighted voting by node operators, not a REST API.

### 1. FTNS Token Management

Complete Web3 integration for FTNS tokens:

```typescript
// Check balance
const balance = await client.ftns.getBalance();
console.log(`Available: ${balance.availableBalance} FTNS`);

// Transfer tokens
const transfer = await client.ftns.transfer({
  toAddress: '0x742d35cc6bf4532c95a0e96a7bdc86c0b3e11888',
  amount: 100,
  note: 'Research collaboration payment'
});

// Purchase tokens with fiat
const purchase = await client.ftns.purchaseTokens({
  amountUsd: 50,
  paymentMethod: 'stripe',
  paymentToken: 'tok_1234567890abcdef'
});
```

### 2. ContentStore (Publish Data with Royalty Tracking)

Upload content through your node's ContentStore and earn 80% of every query that hits it:

```typescript
// Upload a dataset
const result = await client.storage.upload({
  filePath: './climate_data.parquet',
  description: 'NOAA climate observations 2025',
  royaltyRate: 0.05,   // 0.05 FTNS per access
  replicas: 5,
});
console.log(`CID: ${result.cid}`);

// Download by CID
const download = await client.storage.download(result.cid);

// Search semantic shards
const shards = await client.storage.searchShards({
  query: 'climate observations',
  limit: 10,
});
```

### 3. Real-time WebSocket Updates

Get live updates for sessions, tools, and system events:

```typescript
// Connect WebSocket
await client.websocket.connect();

// Subscribe to session progress
client.websocket.subscribeToSession(sessionId, (progress) => {
  console.log(`${progress.progress}% - ${progress.currentStep}`);
});

// Subscribe to balance updates
client.websocket.subscribeToBalanceUpdates((balance) => {
  console.log('Balance updated:', balance.totalBalance, 'FTNS');
});

// Subscribe to safety alerts
client.websocket.subscribeToSafetyAlerts((alert) => {
  console.log('Safety alert:', alert.level, alert.message);
});
```

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
git clone https://github.com/prsm-network/PRSM.git
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
- 🐛 **Issues**: [GitHub Issues](https://github.com/prsm-network/PRSM/issues)
- 💬 **Community**: [Discord](https://discord.gg/prsm)
- 📧 **Email**: [dev@prsm.ai](mailto:dev@prsm.ai)