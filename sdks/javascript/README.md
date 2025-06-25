# PRSM JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for the **Protocol for Recursive Scientific Modeling (PRSM)**, featuring MIT's breakthrough SEAL (Self-Adapting Language Models) technology.

[![npm version](https://badge.fury.io/js/%40prsm%2Fsdk.svg)](https://badge.fury.io/js/%40prsm%2Fsdk)
[![TypeScript](https://img.shields.io/badge/%3C%2F%3E-TypeScript-%230074c1.svg)](https://www.typescriptlang.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🧠 **Complete API Client** - Full access to all PRSM endpoints including NWTN, SEAL, and marketplace
- 🔒 **TypeScript Support** - Comprehensive type definitions for enhanced developer experience
- ⚡ **Real-time WebSocket** - Live session progress updates and streaming responses
- 🛡️ **Error Handling** - Robust error handling with automatic retry logic
- 🔑 **Authentication** - JWT tokens, API keys, and secure session management
- 💰 **FTNS Integration** - Complete token management and Web3 operations
- 🏪 **Marketplace** - Model discovery, rental, and submission capabilities
- 🔧 **Tool Execution** - Discover and execute tools with safety validation
- 📊 **Circuit Breakers** - Built-in safety mechanisms and rate limiting
- 🌐 **Cross-platform** - Works in browsers, Node.js, and React Native

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
  baseUrl: 'https://api.prsm.org'
});

// Submit a research query
const result = await client.query(
  'Analyze the impact of climate change on marine ecosystems',
  {
    domain: 'environmental_science',
    maxIterations: 5,
    includeCitations: true
  }
);

console.log(result.content);
console.log(result.citations);
```

### Advanced NWTN Usage

```typescript
// Submit query with SEAL enhancement
const session = await client.nwtn.submitQuery({
  query: 'Develop a novel approach to protein folding prediction',
  domain: 'biochemistry',
  methodology: 'comprehensive_analysis',
  maxIterations: 8,
  includeCitations: true,
  sealEnhancement: {
    enabled: true,
    autonomousImprovement: true,
    targetLearningGain: 0.20,
    restemMethodology: true
  }
});

// Monitor progress with WebSocket
await client.websocket.connect();
client.websocket.subscribeToSession(session.sessionId, (progress) => {
  console.log(`Progress: ${progress.progress}%`);
  console.log(`Current step: ${progress.currentStep}`);
});

// Wait for completion
const result = await client.nwtn.waitForCompletion(session.sessionId);
console.log('Research completed:', result.results?.summary);
```

## Core Features

### 1. Neural Web of Thought Networks (NWTN)

Submit complex research queries with autonomous reasoning:

```typescript
const session = await client.nwtn.submitQuery({
  query: 'Design a sustainable urban transportation system',
  domain: 'urban_planning',
  methodology: 'comprehensive_analysis',
  maxIterations: 6,
  tools: ['simulation', 'optimization', 'visualization'],
  context: {
    city_size: 'large',
    budget_constraint: 'moderate',
    environmental_priority: 'high'
  }
});
```

### 2. SEAL Technology Integration

Leverage MIT's Self-Adapting Language Models:

```typescript
// Get SEAL performance metrics
const metrics = await client.seal.getMetrics();
console.log('Knowledge incorporation improvement:', 
  metrics.productionMetrics.improvementPercentage);

// Trigger autonomous improvement
const improvement = await client.seal.triggerImprovement({
  domain: 'biomedical_research',
  targetImprovement: 0.25,
  improvementStrategy: 'restem_methodology',
  maxIterations: 15
});
```

### 3. FTNS Token Management

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

### 4. Marketplace Operations

Discover, rent, and submit models:

```typescript
// Browse marketplace
const models = await client.marketplace.browseModels({
  category: 'scientific',
  provider: 'verified',
  minPerformance: 4.0,
  featured: true,
  limit: 20
});

// Rent a model
const rental = await client.marketplace.rentModel('model_123', {
  durationHours: 24,
  maxRequests: 1000
});

// Submit your own model
const submission = await client.marketplace.submitModel({
  name: 'Advanced Climate Model',
  description: 'High-accuracy climate prediction model',
  category: 'scientific',
  modelFile: 'ipfs://QmXXXXXX...',
  pricing: { ftnsPerRequest: 5, revenueShare: 0.7 },
  tags: ['climate', 'prediction', 'environmental']
});
```

### 5. Real-time WebSocket Updates

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