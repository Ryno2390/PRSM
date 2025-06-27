# Node.js Application Integration

Integrate PRSM into your Node.js and TypeScript applications for intelligent AI-powered features.

## 🎯 Overview

This guide covers integrating PRSM into Node.js applications using Express, Next.js, or any Node.js framework. PRSM's JavaScript SDK provides TypeScript support and async/await patterns for seamless integration.

## 📋 Prerequisites

- Node.js 16+ installed
- PRSM instance running (local or remote)
- Basic understanding of async/await patterns

## 🚀 Quick Start

### 1. Installation

```bash
npm install @prsm/sdk
# or
yarn add @prsm/sdk
```

### 2. Basic Setup

```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  baseURL: 'http://localhost:8000',
  apiKey: process.env.PRSM_API_KEY,
  timeout: 30000
});
```

### 3. Simple Query

```javascript
async function askPRSM(prompt) {
  try {
    const response = await client.query({
      prompt: prompt,
      userId: 'user-123',
      sessionId: 'session-456'
    });
    
    return response.data.answer;
  } catch (error) {
    console.error('PRSM Error:', error);
    throw error;
  }
}
```

## 🌐 Express.js Integration

### Basic Express App

```javascript
import express from 'express';
import { PRSMClient } from '@prsm/sdk';

const app = express();
const prsm = new PRSMClient({
  baseURL: process.env.PRSM_URL || 'http://localhost:8000',
  apiKey: process.env.PRSM_API_KEY
});

app.use(express.json());

// AI-powered endpoint
app.post('/api/ai/chat', async (req, res) => {
  try {
    const { message, userId } = req.body;
    
    const response = await prsm.query({
      prompt: message,
      userId: userId,
      context: {
        timestamp: new Date().toISOString(),
        endpoint: '/api/ai/chat'
      }
    });
    
    res.json({
      success: true,
      response: response.data.answer,
      usage: response.data.usage
    });
    
  } catch (error) {
    console.error('AI Chat Error:', error);
    res.status(500).json({
      success: false,
      error: 'AI service temporarily unavailable'
    });
  }
});

// Health check with PRSM status
app.get('/health', async (req, res) => {
  try {
    const prsmHealth = await prsm.health();
    res.json({
      status: 'healthy',
      prsm: prsmHealth.data
    });
  } catch (error) {
    res.status(503).json({
      status: 'degraded',
      error: 'PRSM unavailable'
    });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

### Advanced Express Middleware

```javascript
import rateLimit from 'express-rate-limit';
import { PRSMError, PRSMTimeoutError } from '@prsm/sdk';

// Rate limiting for AI endpoints
const aiRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many AI requests, please try again later'
});

// PRSM error handling middleware
function prsmErrorHandler(error, req, res, next) {
  if (error instanceof PRSMTimeoutError) {
    res.status(504).json({
      error: 'AI service timeout',
      code: 'PRSM_TIMEOUT'
    });
  } else if (error instanceof PRSMError) {
    res.status(502).json({
      error: 'AI service error',
      code: 'PRSM_ERROR',
      details: error.message
    });
  } else {
    next(error);
  }
}

// Apply middleware
app.use('/api/ai', aiRateLimit);
app.use(prsmErrorHandler);
```

## ⚡ Next.js Integration

### API Route

```javascript
// pages/api/ai/chat.js
import { PRSMClient } from '@prsm/sdk';

const prsm = new PRSMClient({
  baseURL: process.env.PRSM_URL,
  apiKey: process.env.PRSM_API_KEY
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const { prompt, userId } = req.body;
    
    const response = await prsm.query({
      prompt,
      userId,
      context: {
        userAgent: req.headers['user-agent'],
        ip: req.ip
      }
    });
    
    res.json({
      answer: response.data.answer,
      confidence: response.data.confidence,
      tokens: response.data.usage.tokens
    });
    
  } catch (error) {
    console.error('PRSM API Error:', error);
    res.status(500).json({ error: 'AI service error' });
  }
}
```

### React Hook for PRSM

```typescript
// hooks/usePRSM.ts
import { useState, useCallback } from 'react';

interface PRSMResponse {
  answer: string;
  confidence: number;
  tokens: number;
}

interface UsePRSMResult {
  query: (prompt: string) => Promise<PRSMResponse>;
  loading: boolean;
  error: string | null;
  lastResponse: PRSMResponse | null;
}

export function usePRSM(): UsePRSMResult {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<PRSMResponse | null>(null);
  
  const query = useCallback(async (prompt: string): Promise<PRSMResponse> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          userId: 'current-user' // Replace with actual user ID
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setLastResponse(data);
      return data;
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { query, loading, error, lastResponse };
}
```

## 🔄 Streaming Integration

### Server-Sent Events

```javascript
// Streaming endpoint
app.get('/api/ai/stream', async (req, res) => {
  const { prompt, userId } = req.query;
  
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*'
  });
  
  try {
    const stream = await prsm.streamQuery({
      prompt: prompt,
      userId: userId
    });
    
    stream.on('data', (chunk) => {
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    });
    
    stream.on('end', () => {
      res.write('data: [DONE]\n\n');
      res.end();
    });
    
    stream.on('error', (error) => {
      res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
      res.end();
    });
    
  } catch (error) {
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
});
```

### WebSocket Integration

```javascript
import WebSocket from 'ws';

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  ws.on('message', async (message) => {
    try {
      const { prompt, userId } = JSON.parse(message);
      
      // Stream response through WebSocket
      const stream = await prsm.streamQuery({ prompt, userId });
      
      stream.on('data', (chunk) => {
        ws.send(JSON.stringify({
          type: 'data',
          content: chunk
        }));
      });
      
      stream.on('end', () => {
        ws.send(JSON.stringify({
          type: 'complete'
        }));
      });
      
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        message: error.message
      }));
    }
  });
});
```

## 📊 Background Processing

### Bull Queue Integration

```javascript
import Bull from 'bull';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);
const aiQueue = new Bull('AI Processing', { redis });

// Add job to queue
app.post('/api/ai/process', async (req, res) => {
  const { prompt, userId, priority = 'normal' } = req.body;
  
  const job = await aiQueue.add('process-ai-request', {
    prompt,
    userId,
    timestamp: Date.now()
  }, {
    priority: priority === 'high' ? 10 : 1,
    attempts: 3,
    backoff: 'exponential'
  });
  
  res.json({ jobId: job.id });
});

// Process jobs
aiQueue.process('process-ai-request', async (job) => {
  const { prompt, userId } = job.data;
  
  try {
    const response = await prsm.query({
      prompt,
      userId,
      priority: job.opts.priority > 5 ? 'high' : 'normal'
    });
    
    // Store result or send notification
    await saveResult(userId, response.data);
    await sendNotification(userId, 'AI processing complete');
    
    return response.data;
    
  } catch (error) {
    console.error('Background AI processing failed:', error);
    throw error;
  }
});

// Job status endpoint
app.get('/api/ai/status/:jobId', async (req, res) => {
  const job = await aiQueue.getJob(req.params.jobId);
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  res.json({
    status: await job.getState(),
    progress: job.progress(),
    result: job.returnvalue,
    error: job.failedReason
  });
});
```

## 🔐 Authentication & Security

### JWT Integration

```javascript
import jwt from 'jsonwebtoken';

// Middleware to extract user from JWT
function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// Protected AI endpoint
app.post('/api/ai/chat', authMiddleware, async (req, res) => {
  const { message } = req.body;
  const userId = req.user.id;
  
  try {
    const response = await prsm.query({
      prompt: message,
      userId: userId,
      context: {
        userRole: req.user.role,
        permissions: req.user.permissions
      }
    });
    
    res.json({ response: response.data.answer });
    
  } catch (error) {
    res.status(500).json({ error: 'AI service error' });
  }
});
```

## 📊 Monitoring & Logging

### Structured Logging

```javascript
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'logs/prsm-integration.log' }),
    new winston.transports.Console()
  ]
});

// Log PRSM requests
async function loggedPRSMQuery(prompt, userId, context = {}) {
  const startTime = Date.now();
  
  try {
    logger.info('PRSM query started', {
      userId,
      promptLength: prompt.length,
      context
    });
    
    const response = await prsm.query({ prompt, userId, context });
    
    logger.info('PRSM query completed', {
      userId,
      duration: Date.now() - startTime,
      tokens: response.data.usage?.tokens,
      confidence: response.data.confidence
    });
    
    return response;
    
  } catch (error) {
    logger.error('PRSM query failed', {
      userId,
      duration: Date.now() - startTime,
      error: error.message,
      stack: error.stack
    });
    throw error;
  }
}
```

### Prometheus Metrics

```javascript
import promClient from 'prom-client';

// Define metrics
const prsmQueryDuration = new promClient.Histogram({
  name: 'prsm_query_duration_seconds',
  help: 'Duration of PRSM queries',
  labelNames: ['user_id', 'status']
});

const prsmQueryCounter = new promClient.Counter({
  name: 'prsm_queries_total',
  help: 'Total number of PRSM queries',
  labelNames: ['user_id', 'status']
});

// Instrument PRSM calls
async function instrumentedQuery(prompt, userId) {
  const timer = prsmQueryDuration.startTimer({ user_id: userId });
  
  try {
    const response = await prsm.query({ prompt, userId });
    
    timer({ status: 'success' });
    prsmQueryCounter.inc({ user_id: userId, status: 'success' });
    
    return response;
    
  } catch (error) {
    timer({ status: 'error' });
    prsmQueryCounter.inc({ user_id: userId, status: 'error' });
    throw error;
  }
}

// Metrics endpoint
app.get('/metrics', (req, res) => {
  res.set('Content-Type', promClient.register.contentType);
  res.end(promClient.register.metrics());
});
```

## 🧪 Testing

### Jest Integration Tests

```javascript
// __tests__/prsm-integration.test.js
import { PRSMClient } from '@prsm/sdk';

describe('PRSM Integration', () => {
  let client;
  
  beforeAll(() => {
    client = new PRSMClient({
      baseURL: process.env.TEST_PRSM_URL || 'http://localhost:8000',
      apiKey: 'test-api-key'
    });
  });
  
  test('should respond to simple query', async () => {
    const response = await client.query({
      prompt: 'What is 2+2?',
      userId: 'test-user'
    });
    
    expect(response.data.answer).toBeDefined();
    expect(typeof response.data.answer).toBe('string');
  });
  
  test('should handle errors gracefully', async () => {
    await expect(client.query({
      prompt: '', // Invalid empty prompt
      userId: 'test-user'
    })).rejects.toThrow();
  });
  
  test('should include usage information', async () => {
    const response = await client.query({
      prompt: 'Test query',
      userId: 'test-user'
    });
    
    expect(response.data.usage).toBeDefined();
    expect(response.data.usage.tokens).toBeGreaterThan(0);
  });
});
```

## 🚀 Deployment Considerations

### Environment Configuration

```javascript
// config/production.js
export default {
  prsm: {
    baseURL: process.env.PRSM_URL,
    apiKey: process.env.PRSM_API_KEY,
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
    
    // Connection pooling
    maxConnections: 100,
    keepAlive: true,
    
    // Rate limiting
    rateLimit: {
      requests: 1000,
      window: '1h'
    }
  },
  
  logging: {
    level: 'info',
    format: 'json'
  },
  
  metrics: {
    enabled: true,
    endpoint: '/metrics'
  }
};
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["npm", "start"]
```

## 📋 Best Practices

### Error Handling

```javascript
class PRSMService {
  constructor(options) {
    this.client = new PRSMClient(options);
    this.circuitBreaker = new CircuitBreaker(this.query.bind(this), {
      timeout: 30000,
      errorThresholdPercentage: 50,
      resetTimeout: 60000
    });
  }
  
  async query(prompt, userId, options = {}) {
    try {
      return await this.client.query({
        prompt,
        userId,
        ...options
      });
    } catch (error) {
      // Log error details
      console.error('PRSM Service Error:', {
        error: error.message,
        userId,
        promptLength: prompt.length,
        timestamp: new Date().toISOString()
      });
      
      // Return graceful fallback
      return {
        data: {
          answer: 'I apologize, but I cannot process your request right now. Please try again later.',
          confidence: 0,
          fallback: true
        }
      };
    }
  }
}
```

### Caching Strategy

```javascript
import NodeCache from 'node-cache';

const cache = new NodeCache({ stdTTL: 300 }); // 5 minutes

async function cachedQuery(prompt, userId) {
  const cacheKey = `prsm:${userId}:${hashPrompt(prompt)}`;
  
  // Check cache first
  const cached = cache.get(cacheKey);
  if (cached) {
    return cached;
  }
  
  // Query PRSM
  const response = await prsm.query({ prompt, userId });
  
  // Cache if confidence is high
  if (response.data.confidence > 0.8) {
    cache.set(cacheKey, response);
  }
  
  return response;
}
```

## 🔗 Next Steps

- [Express.js Advanced Patterns](../../API_REFERENCE.md#express-integration)
- [Next.js Production Deployment](../../PRODUCTION_OPERATIONS_MANUAL.md#deployment)
- [TypeScript Type Definitions](../../SDK_DOCUMENTATION.md#typescript)
- [Performance Optimization](../../performance/PERFORMANCE_ASSESSMENT.md)

## 🆘 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection timeout | Network issues or PRSM down | Implement retry logic and circuit breaker |
| High memory usage | Not properly closing connections | Use connection pooling |
| Slow responses | No caching | Implement response caching |
| Authentication errors | Invalid API key | Check environment variables |

### Debug Mode

```javascript
const client = new PRSMClient({
  baseURL: process.env.PRSM_URL,
  apiKey: process.env.PRSM_API_KEY,
  debug: process.env.NODE_ENV === 'development',
  logLevel: 'debug'
});
```

---

**Need help?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).