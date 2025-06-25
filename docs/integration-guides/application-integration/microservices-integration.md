# Microservices Architecture Integration

Integrate PRSM into enterprise microservices architectures with service mesh, distributed tracing, and scalable patterns.

## 🎯 Overview

This guide covers integrating PRSM into microservices architectures, including service discovery, load balancing, distributed tracing, and inter-service communication patterns.

## 🏗️ Architecture Patterns

### Service Mesh Integration

```yaml
# istio-prsm-gateway.yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: prsm-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - prsm-api.example.com
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: prsm-routing
spec:
  hosts:
  - prsm-api.example.com
  gateways:
  - prsm-gateway
  http:
  - match:
    - uri:
        prefix: "/api/v1/"
    route:
    - destination:
        host: prsm-api-service
        port:
          number: 8000
      weight: 90
    - destination:
        host: prsm-api-service-canary
        port:
          number: 8000
      weight: 10
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 30s
```

### API Gateway Pattern

```javascript
// api-gateway/prsm-proxy.js
import express from 'express';
import httpProxy from 'http-proxy-middleware';
import { createProxyMiddleware } from 'http-proxy-middleware';
import rateLimit from 'express-rate-limit';

const app = express();

// Rate limiting per service
const prsmRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  keyGenerator: (req) => `${req.ip}:prsm`,
  message: 'Too many PRSM requests'
});

// Load balancer for PRSM instances
const prsmProxy = createProxyMiddleware({
  target: 'http://prsm-service:8000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/ai': '/api/v1'
  },
  router: {
    // Route to different PRSM instances based on load
    '/api/ai/heavy': 'http://prsm-gpu-service:8000',
    '/api/ai/fast': 'http://prsm-cpu-service:8000'
  },
  onProxyReq: (proxyReq, req, res) => {
    // Add correlation ID for tracing
    proxyReq.setHeader('X-Correlation-ID', req.headers['x-correlation-id'] || generateCorrelationId());
    proxyReq.setHeader('X-Source-Service', req.headers['x-source-service'] || 'api-gateway');
  },
  onError: (err, req, res) => {
    console.error('PRSM Proxy Error:', err);
    res.status(502).json({
      error: 'AI service temporarily unavailable',
      correlationId: req.headers['x-correlation-id']
    });
  }
});

app.use('/api/ai', prsmRateLimit, prsmProxy);
app.listen(3000);
```

## 🔄 Service Discovery

### Consul Integration

```javascript
// services/prsm-client.js
import Consul from 'consul';
import { PRSMClient } from '@prsm/sdk';

class PRSMServiceClient {
  constructor() {
    this.consul = new Consul();
    this.clients = new Map();
    this.healthCheck();
  }
  
  async discoverPRSMServices() {
    try {
      const services = await this.consul.health.service({
        service: 'prsm-api',
        passing: true
      });
      
      const instances = services.map(service => ({
        id: service.Service.ID,
        address: service.Service.Address,
        port: service.Service.Port,
        tags: service.Service.Tags,
        weight: this.calculateWeight(service.Service.Tags)
      }));
      
      this.updateClientPool(instances);
      return instances;
      
    } catch (error) {
      console.error('Service discovery failed:', error);
      return [];
    }
  }
  
  updateClientPool(instances) {
    // Remove old clients
    for (const [id, client] of this.clients) {
      if (!instances.find(i => i.id === id)) {
        this.clients.delete(id);
      }
    }
    
    // Add new clients
    instances.forEach(instance => {
      if (!this.clients.has(instance.id)) {
        this.clients.set(instance.id, new PRSMClient({
          baseURL: `http://${instance.address}:${instance.port}`,
          timeout: 30000,
          retries: 3
        }));
      }
    });
  }
  
  async query(prompt, userId, options = {}) {
    const instances = Array.from(this.clients.entries());
    if (instances.length === 0) {
      throw new Error('No healthy PRSM instances available');
    }
    
    // Weighted round-robin selection
    const [id, client] = this.selectInstance(instances, options);
    
    try {
      return await client.query({ prompt, userId, ...options });
    } catch (error) {
      // Fallback to another instance
      const fallbackInstances = instances.filter(([fallbackId]) => fallbackId !== id);
      if (fallbackInstances.length > 0) {
        const [, fallbackClient] = fallbackInstances[0];
        return await fallbackClient.query({ prompt, userId, ...options });
      }
      throw error;
    }
  }
  
  selectInstance(instances, options) {
    // Simple round-robin for now
    const index = Math.floor(Math.random() * instances.length);
    return instances[index];
  }
  
  async healthCheck() {
    setInterval(async () => {
      await this.discoverPRSMServices();
    }, 30000); // Check every 30 seconds
  }
}

export default new PRSMServiceClient();
```

### Kubernetes Service Discovery

```yaml
# k8s/prsm-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: prsm-api-service
  labels:
    app: prsm-api
    version: v1
spec:
  selector:
    app: prsm-api
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
        version: v1
    spec:
      containers:
      - name: prsm-api
        image: prsm/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SERVICE_NAME
          value: "prsm-api"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 📊 Distributed Tracing

### Jaeger Integration

```javascript
// tracing/jaeger-setup.js
import { initTracer } from 'jaeger-client';
import opentracing from 'opentracing';

const config = {
  serviceName: 'prsm-integration-service',
  reporter: {
    logSpans: true,
    agentHost: process.env.JAEGER_AGENT_HOST || 'localhost',
    agentPort: process.env.JAEGER_AGENT_PORT || 6832,
  },
  sampler: {
    type: 'const',
    param: 1,
  },
};

const tracer = initTracer(config);
opentracing.initGlobalTracer(tracer);

export { tracer };
```

```javascript
// services/traced-prsm-client.js
import opentracing from 'opentracing';
import { PRSMClient } from '@prsm/sdk';

class TracedPRSMClient {
  constructor(options) {
    this.client = new PRSMClient(options);
    this.tracer = opentracing.globalTracer();
  }
  
  async query(prompt, userId, parentSpan = null) {
    const span = this.tracer.startSpan('prsm.query', {
      childOf: parentSpan,
      tags: {
        'component': 'prsm-client',
        'user.id': userId,
        'prompt.length': prompt.length,
        'service.name': 'prsm-api'
      }
    });
    
    try {
      span.log({
        event: 'query.start',
        message: 'Starting PRSM query',
        userId: userId
      });
      
      const response = await this.client.query({
        prompt,
        userId,
        context: {
          traceId: span.context().traceId,
          spanId: span.context().spanId
        }
      });
      
      span.setTag('response.tokens', response.data.usage?.tokens);
      span.setTag('response.confidence', response.data.confidence);
      span.setTag('http.status_code', 200);
      
      span.log({
        event: 'query.complete',
        message: 'PRSM query completed successfully'
      });
      
      return response;
      
    } catch (error) {
      span.setTag('error', true);
      span.setTag('http.status_code', error.status || 500);
      span.log({
        event: 'error',
        'error.object': error,
        'error.message': error.message,
        stack: error.stack
      });
      
      throw error;
    } finally {
      span.finish();
    }
  }
}

export default TracedPRSMClient;
```

### OpenTelemetry Integration

```javascript
// telemetry/otel-setup.js
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';

const jaegerExporter = new JaegerExporter({
  endpoint: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
});

const prometheusExporter = new PrometheusExporter({
  port: 9090,
}, () => {
  console.log('Prometheus metrics server started on port 9090');
});

const sdk = new NodeSDK({
  traceExporter: jaegerExporter,
  metricReader: prometheusExporter,
  instrumentations: [getNodeAutoInstrumentations({
    '@opentelemetry/instrumentation-fs': {
      enabled: false,
    },
  })]
});

sdk.start();
console.log('OpenTelemetry started successfully');
```

## 🔄 Inter-Service Communication

### Event-Driven Architecture

```javascript
// events/prsm-event-handler.js
import { EventEmitter } from 'events';
import Redis from 'ioredis';

class PRSMEventHandler extends EventEmitter {
  constructor() {
    super();
    this.redis = new Redis(process.env.REDIS_URL);
    this.subscriber = new Redis(process.env.REDIS_URL);
    this.setupSubscriptions();
  }
  
  setupSubscriptions() {
    this.subscriber.subscribe('prsm:query:completed', 'prsm:query:failed');
    
    this.subscriber.on('message', (channel, message) => {
      try {
        const data = JSON.parse(message);
        this.handleEvent(channel, data);
      } catch (error) {
        console.error('Failed to parse event message:', error);
      }
    });
  }
  
  handleEvent(channel, data) {
    switch (channel) {
      case 'prsm:query:completed':
        this.emit('query:completed', data);
        this.updateUserMetrics(data.userId, data.usage);
        this.notifyDownstreamServices(data);
        break;
        
      case 'prsm:query:failed':
        this.emit('query:failed', data);
        this.logFailure(data);
        this.triggerFallback(data);
        break;
    }
  }
  
  async publishQueryEvent(type, data) {
    const event = {
      type,
      timestamp: new Date().toISOString(),
      correlationId: data.correlationId || this.generateCorrelationId(),
      ...data
    };
    
    await this.redis.publish(`prsm:query:${type}`, JSON.stringify(event));
  }
  
  async updateUserMetrics(userId, usage) {
    const key = `user:${userId}:metrics`;
    await this.redis.hincrby(key, 'total_queries', 1);
    await this.redis.hincrby(key, 'total_tokens', usage.tokens || 0);
    await this.redis.expire(key, 86400); // 24 hours
  }
  
  async triggerFallback(data) {
    // Trigger fallback service
    await this.redis.publish('fallback:ai:request', JSON.stringify({
      originalRequest: data.request,
      failureReason: data.error,
      correlationId: data.correlationId
    }));
  }
}

export default new PRSMEventHandler();
```

### Message Queue Integration

```javascript
// queues/prsm-queue-processor.js
import Bull from 'bull';
import Redis from 'ioredis';
import { PRSMClient } from '@prsm/sdk';

const redis = new Redis(process.env.REDIS_URL);
const prsmQueue = new Bull('PRSM Processing', { redis });
const prsmClient = new PRSMClient({
  baseURL: process.env.PRSM_URL,
  apiKey: process.env.PRSM_API_KEY
});

// Define job types
prsmQueue.process('priority-query', 5, async (job) => {
  return await processPriorityQuery(job.data);
});

prsmQueue.process('batch-query', 10, async (job) => {
  return await processBatchQuery(job.data);
});

prsmQueue.process('scheduled-query', 2, async (job) => {
  return await processScheduledQuery(job.data);
});

async function processPriorityQuery(data) {
  const { prompt, userId, requestId } = data;
  
  try {
    const response = await prsmClient.query({
      prompt,
      userId,
      priority: 'high',
      context: {
        requestId,
        queueType: 'priority'
      }
    });
    
    // Notify requesting service immediately
    await notifyService(data.callbackUrl, {
      requestId,
      status: 'completed',
      response: response.data
    });
    
    return response.data;
    
  } catch (error) {
    await notifyService(data.callbackUrl, {
      requestId,
      status: 'failed',
      error: error.message
    });
    throw error;
  }
}

async function processBatchQuery(data) {
  const { queries, batchId } = data;
  const results = [];
  
  for (const query of queries) {
    try {
      const response = await prsmClient.query({
        ...query,
        context: {
          batchId,
          queueType: 'batch'
        }
      });
      results.push({
        id: query.id,
        status: 'success',
        response: response.data
      });
    } catch (error) {
      results.push({
        id: query.id,
        status: 'error',
        error: error.message
      });
    }
  }
  
  // Store batch results
  await redis.setex(`batch:${batchId}:results`, 3600, JSON.stringify(results));
  
  return results;
}

// Queue monitoring
prsmQueue.on('completed', (job, result) => {
  console.log(`Job ${job.id} completed successfully`);
  // Update metrics
  updateJobMetrics(job.opts.type, 'completed', job.processedOn - job.timestamp);
});

prsmQueue.on('failed', (job, err) => {
  console.error(`Job ${job.id} failed:`, err);
  updateJobMetrics(job.opts.type, 'failed');
});

async function notifyService(callbackUrl, data) {
  try {
    await fetch(callbackUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  } catch (error) {
    console.error('Failed to notify service:', error);
  }
}
```

## 🔐 Security & Authentication

### OAuth2 Service-to-Service

```javascript
// auth/oauth2-client.js
import { AuthenticationClient } from 'auth0';

class ServiceAuthClient {
  constructor() {
    this.auth0 = new AuthenticationClient({
      domain: process.env.AUTH0_DOMAIN,
      clientId: process.env.AUTH0_CLIENT_ID,
      clientSecret: process.env.AUTH0_CLIENT_SECRET
    });
    this.tokens = new Map();
  }
  
  async getServiceToken(audience) {
    const cacheKey = `service:${audience}`;
    const cached = this.tokens.get(cacheKey);
    
    if (cached && cached.expiresAt > Date.now()) {
      return cached.token;
    }
    
    try {
      const response = await this.auth0.clientCredentialsGrant({
        audience: audience,
        scope: 'read:ai write:ai'
      });
      
      const expiresAt = Date.now() + (response.expires_in * 1000) - 60000; // 1 min buffer
      
      this.tokens.set(cacheKey, {
        token: response.access_token,
        expiresAt
      });
      
      return response.access_token;
      
    } catch (error) {
      console.error('Failed to get service token:', error);
      throw error;
    }
  }
}

export default new ServiceAuthClient();
```

### mTLS Configuration

```javascript
// security/mtls-client.js
import https from 'https';
import fs from 'fs';
import { PRSMClient } from '@prsm/sdk';

class SecurePRSMClient extends PRSMClient {
  constructor(options) {
    const httpsAgent = new https.Agent({
      cert: fs.readFileSync(process.env.CLIENT_CERT_PATH),
      key: fs.readFileSync(process.env.CLIENT_KEY_PATH),
      ca: fs.readFileSync(process.env.CA_CERT_PATH),
      rejectUnauthorized: true,
      checkServerIdentity: (host, cert) => {
        // Custom server identity verification
        return undefined; // No error = valid
      }
    });
    
    super({
      ...options,
      httpsAgent,
      headers: {
        ...options.headers,
        'X-Client-Cert-Subject': this.extractCertSubject(),
        'X-Service-Name': process.env.SERVICE_NAME
      }
    });
  }
  
  extractCertSubject() {
    const cert = fs.readFileSync(process.env.CLIENT_CERT_PATH);
    // Extract subject from certificate
    return 'CN=prsm-service,O=Example Corp';
  }
}
```

## 📊 Monitoring & Observability

### Metrics Collection

```javascript
// monitoring/prsm-metrics.js
import promClient from 'prom-client';

// Define custom metrics
const prsmQueryDuration = new promClient.Histogram({
  name: 'prsm_query_duration_seconds',
  help: 'Duration of PRSM queries',
  labelNames: ['service', 'method', 'status', 'user_tier'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30]
});

const prsmQueryCounter = new promClient.Counter({
  name: 'prsm_queries_total',
  help: 'Total number of PRSM queries',
  labelNames: ['service', 'method', 'status', 'user_tier']
});

const prsmActiveConnections = new promClient.Gauge({
  name: 'prsm_active_connections',
  help: 'Number of active PRSM connections',
  labelNames: ['service', 'instance']
});

const prsmTokenUsage = new promClient.Counter({
  name: 'prsm_tokens_used_total',
  help: 'Total tokens consumed',
  labelNames: ['service', 'user_tier', 'model']
});

class PRSMMetrics {
  constructor(serviceName) {
    this.serviceName = serviceName;
    this.activeConnections = 0;
  }
  
  recordQuery(method, status, duration, userTier, tokens = 0, model = 'default') {
    const labels = {
      service: this.serviceName,
      method,
      status,
      user_tier: userTier
    };
    
    prsmQueryDuration.observe(labels, duration);
    prsmQueryCounter.inc(labels);
    
    if (tokens > 0) {
      prsmTokenUsage.inc({
        service: this.serviceName,
        user_tier: userTier,
        model
      }, tokens);
    }
  }
  
  incrementConnections() {
    this.activeConnections++;
    prsmActiveConnections.set({
      service: this.serviceName,
      instance: process.env.INSTANCE_ID || 'unknown'
    }, this.activeConnections);
  }
  
  decrementConnections() {
    this.activeConnections--;
    prsmActiveConnections.set({
      service: this.serviceName,
      instance: process.env.INSTANCE_ID || 'unknown'
    }, this.activeConnections);
  }
}

export default PRSMMetrics;
```

### Health Checks

```javascript
// health/prsm-health-check.js
class PRSMHealthCheck {
  constructor(prsmClient) {
    this.prsmClient = prsmClient;
    this.lastHealthCheck = null;
    this.healthStatus = 'unknown';
  }
  
  async checkHealth() {
    try {
      const start = Date.now();
      const response = await this.prsmClient.health();
      const duration = Date.now() - start;
      
      this.lastHealthCheck = new Date();
      this.healthStatus = response.status === 'healthy' && duration < 5000 ? 'healthy' : 'degraded';
      
      return {
        status: this.healthStatus,
        responseTime: duration,
        details: response.data,
        timestamp: this.lastHealthCheck
      };
      
    } catch (error) {
      this.healthStatus = 'unhealthy';
      this.lastHealthCheck = new Date();
      
      return {
        status: 'unhealthy',
        error: error.message,
        timestamp: this.lastHealthCheck
      };
    }
  }
  
  async deepHealthCheck() {
    const healthResult = await this.checkHealth();
    
    if (healthResult.status !== 'healthy') {
      return healthResult;
    }
    
    // Test actual functionality
    try {
      const testQuery = await this.prsmClient.query({
        prompt: 'Health check test query',
        userId: 'health-check',
        context: { healthCheck: true }
      });
      
      return {
        ...healthResult,
        functionalTest: 'passed',
        testResponse: testQuery.data
      };
      
    } catch (error) {
      return {
        ...healthResult,
        status: 'degraded',
        functionalTest: 'failed',
        testError: error.message
      };
    }
  }
}

export default PRSMHealthCheck;
```

## 🧪 Testing Strategies

### Contract Testing

```javascript
// tests/contract/prsm-contract.test.js
import { Pact } from '@pact-foundation/pact';
import { PRSMClient } from '@prsm/sdk';

describe('PRSM API Contract', () => {
  const provider = new Pact({
    consumer: 'microservice-consumer',
    provider: 'prsm-api',
    port: 1234,
    log: 'logs/pact.log',
    dir: 'pacts',
    logLevel: 'INFO'
  });
  
  beforeAll(() => provider.setup());
  afterAll(() => provider.finalize());
  afterEach(() => provider.verify());
  
  test('should handle query request', async () => {
    await provider.addInteraction({
      state: 'PRSM is available',
      uponReceiving: 'a query request',
      withRequest: {
        method: 'POST',
        path: '/api/v1/query',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-token'
        },
        body: {
          prompt: 'Test query',
          userId: 'test-user',
          context: {}
        }
      },
      willRespondWith: {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        },
        body: {
          data: {
            answer: 'Test response',
            confidence: 0.95,
            usage: {
              tokens: 50
            }
          }
        }
      }
    });
    
    const client = new PRSMClient({
      baseURL: 'http://localhost:1234',
      apiKey: 'test-token'
    });
    
    const response = await client.query({
      prompt: 'Test query',
      userId: 'test-user'
    });
    
    expect(response.data.answer).toBe('Test response');
    expect(response.data.confidence).toBe(0.95);
  });
});
```

### Integration Testing

```javascript
// tests/integration/microservices.test.js
import { spawn } from 'child_process';
import Docker from 'dockerode';
import { PRSMClient } from '@prsm/sdk';

describe('Microservices Integration', () => {
  let docker;
  let prsmContainer;
  let redisContainer;
  
  beforeAll(async () => {
    docker = new Docker();
    
    // Start Redis
    redisContainer = await docker.createContainer({
      Image: 'redis:alpine',
      ExposedPorts: { '6379/tcp': {} },
      HostConfig: {
        PortBindings: { '6379/tcp': [{ HostPort: '6380' }] }
      }
    });
    await redisContainer.start();
    
    // Start PRSM
    prsmContainer = await docker.createContainer({
      Image: 'prsm/api:test',
      ExposedPorts: { '8000/tcp': {} },
      HostConfig: {
        PortBindings: { '8000/tcp': [{ HostPort: '8001' }] }
      },
      Env: [
        'REDIS_URL=redis://localhost:6380',
        'API_KEY=test-key'
      ]
    });
    await prsmContainer.start();
    
    // Wait for services to be ready
    await new Promise(resolve => setTimeout(resolve, 5000));
  }, 30000);
  
  afterAll(async () => {
    if (prsmContainer) await prsmContainer.remove({ force: true });
    if (redisContainer) await redisContainer.remove({ force: true });
  });
  
  test('should handle end-to-end request flow', async () => {
    const client = new PRSMClient({
      baseURL: 'http://localhost:8001',
      apiKey: 'test-key'
    });
    
    const response = await client.query({
      prompt: 'Integration test query',
      userId: 'integration-test-user'
    });
    
    expect(response.data.answer).toBeDefined();
    expect(response.data.usage.tokens).toBeGreaterThan(0);
  });
});
```

## 🚀 Deployment Patterns

### Blue-Green Deployment

```yaml
# k8s/blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: prsm-api-rollout
spec:
  replicas: 5
  strategy:
    blueGreen:
      activeService: prsm-api-active
      previewService: prsm-api-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: prsm-api-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: prsm-api-active
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
    spec:
      containers:
      - name: prsm-api
        image: prsm/api:latest
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Canary Deployment

```yaml
# k8s/canary-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: prsm-api-canary
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 1m}
      - setWeight: 30
      - pause: {duration: 2m}
      - setWeight: 50
      - pause: {duration: 3m}
      - setWeight: 100
      analysis:
        templates:
        - templateName: success-rate
        - templateName: latency
        args:
        - name: service-name
          value: prsm-api
      trafficRouting:
        istio:
          virtualService:
            name: prsm-api-vs
          destinationRule:
            name: prsm-api-dr
            canarySubsetName: canary
            stableSubsetName: stable
  selector:
    matchLabels:
      app: prsm-api
  template:
    metadata:
      labels:
        app: prsm-api
    spec:
      containers:
      - name: prsm-api
        image: prsm/api:latest
```

## 📋 Best Practices

### Configuration Management

```javascript
// config/microservices-config.js
import convict from 'convict';

const config = convict({
  env: {
    doc: 'The application environment',
    format: ['production', 'development', 'test'],
    default: 'development',
    env: 'NODE_ENV'
  },
  prsm: {
    baseURL: {
      doc: 'PRSM API base URL',
      format: 'url',
      default: 'http://localhost:8000',
      env: 'PRSM_BASE_URL'
    },
    timeout: {
      doc: 'Request timeout in milliseconds',
      format: 'int',
      default: 30000,
      env: 'PRSM_TIMEOUT'
    },
    retries: {
      doc: 'Number of retry attempts',
      format: 'int',
      default: 3,
      env: 'PRSM_RETRIES'
    }
  },
  monitoring: {
    enabled: {
      doc: 'Enable monitoring',
      format: Boolean,
      default: true,
      env: 'MONITORING_ENABLED'
    },
    metricsPort: {
      doc: 'Metrics server port',
      format: 'port',
      default: 9090,
      env: 'METRICS_PORT'
    }
  }
});

config.validate({ allowed: 'strict' });

export default config;
```

### Circuit Breaker Pattern

```javascript
// patterns/circuit-breaker.js
import CircuitBreaker from 'opossum';

class PRSMCircuitBreaker {
  constructor(prsmClient) {
    this.client = prsmClient;
    this.breaker = new CircuitBreaker(this.query.bind(this), {
      timeout: 30000,
      errorThresholdPercentage: 50,
      resetTimeout: 60000,
      rollingCountTimeout: 10000,
      rollingCountBuckets: 10
    });
    
    this.setupEventHandlers();
  }
  
  setupEventHandlers() {
    this.breaker.on('open', () => {
      console.warn('PRSM Circuit breaker opened');
    });
    
    this.breaker.on('halfOpen', () => {
      console.info('PRSM Circuit breaker half-open');
    });
    
    this.breaker.on('close', () => {
      console.info('PRSM Circuit breaker closed');
    });
    
    this.breaker.fallback(() => ({
      data: {
        answer: 'Service temporarily unavailable. Please try again later.',
        confidence: 0,
        fallback: true
      }
    }));
  }
  
  async query(options) {
    return await this.client.query(options);
  }
  
  async execute(options) {
    return await this.breaker.fire(options);
  }
  
  getStatus() {
    return {
      state: this.breaker.stats.state,
      failures: this.breaker.stats.failures,
      successes: this.breaker.stats.successes,
      rejectCount: this.breaker.stats.rejectCount
    };
  }
}

export default PRSMCircuitBreaker;
```

## 📋 Next Steps

- [Kubernetes Production Deployment](../platform-integration/kubernetes-integration.md)
- [Service Mesh Configuration](./service-mesh-integration.md)
- [Distributed Tracing Setup](../devops-integration/monitoring-integration.md)
- [Performance Testing](../devops-integration/testing-integration.md)

---

**Need help with microservices integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).