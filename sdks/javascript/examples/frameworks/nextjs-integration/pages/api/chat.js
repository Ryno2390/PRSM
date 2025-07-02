// PRSM JavaScript SDK - Next.js API Route Integration
// This example demonstrates how to integrate PRSM with Next.js API routes
// for building modern AI-powered web applications.

import { PRSMClient } from '@prsm/sdk';

// Initialize PRSM client (in production, use environment variables)
const prsm = new PRSMClient({
  apiKey: process.env.PRSM_API_KEY,
  baseURL: process.env.PRSM_BASE_URL || 'https://api.prsm.ai'
});

// Rate limiting storage (in production, use Redis)
const rateLimitStore = new Map();

// Helper function for rate limiting
function checkRateLimit(clientId, limit = 10, windowMs = 60000) {
  const now = Date.now();
  const key = clientId;
  
  if (!rateLimitStore.has(key)) {
    rateLimitStore.set(key, []);
  }
  
  const requests = rateLimitStore.get(key);
  
  // Remove old requests outside the window
  const validRequests = requests.filter(time => now - time < windowMs);
  
  if (validRequests.length >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetTime: Math.min(...validRequests) + windowMs
    };
  }
  
  validRequests.push(now);
  rateLimitStore.set(key, validRequests);
  
  return {
    allowed: true,
    remaining: limit - validRequests.length,
    resetTime: now + windowMs
  };
}

// Helper function to get client identifier
function getClientId(req) {
  return req.headers['x-forwarded-for'] || 
         req.connection.remoteAddress || 
         req.socket.remoteAddress ||
         (req.connection.socket ? req.connection.socket.remoteAddress : null) ||
         'unknown';
}

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed',
      message: 'Only POST requests are supported'
    });
  }

  // Check rate limiting
  const clientId = getClientId(req);
  const rateLimit = checkRateLimit(clientId);
  
  // Add rate limit headers
  res.setHeader('X-RateLimit-Limit', '10');
  res.setHeader('X-RateLimit-Remaining', rateLimit.remaining.toString());
  res.setHeader('X-RateLimit-Reset', Math.ceil(rateLimit.resetTime / 1000).toString());
  
  if (!rateLimit.allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      message: 'Too many requests. Please try again later.',
      retryAfter: Math.ceil((rateLimit.resetTime - Date.now()) / 1000)
    });
  }

  try {
    // Validate request body
    const { message, model, stream = false, options = {} } = req.body;
    
    if (!message || typeof message !== 'string') {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Message is required and must be a string'
      });
    }

    if (message.length > 10000) {
      return res.status(400).json({
        error: 'Message too long',
        message: 'Message must be less than 10,000 characters'
      });
    }

    // Set default options
    const queryOptions = {
      model: model || 'gpt-4',
      prompt: message,
      maxTokens: options.maxTokens || 500,
      temperature: options.temperature || 0.7,
      ...options
    };

    // Add request metadata for logging
    const requestId = req.headers['x-request-id'] || `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    console.log(`[${requestId}] Processing chat request`, {
      model: queryOptions.model,
      messageLength: message.length,
      stream,
      clientId
    });

    // Handle streaming responses
    if (stream) {
      // Set headers for Server-Sent Events
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      });

      try {
        // Stream the response
        for await (const chunk of prsm.models.stream(queryOptions)) {
          const data = JSON.stringify({
            content: chunk.content,
            model: chunk.model,
            requestId
          });
          
          res.write(`data: ${data}\n\n`);
        }

        // Send completion signal
        res.write(`data: ${JSON.stringify({ done: true, requestId })}\n\n`);
        res.end();

      } catch (streamError) {
        console.error(`[${requestId}] Stream error:`, streamError);
        
        const errorData = JSON.stringify({
          error: 'Stream error',
          message: streamError.message || 'An error occurred while streaming',
          requestId
        });
        
        res.write(`data: ${errorData}\n\n`);
        res.end();
      }

    } else {
      // Handle regular (non-streaming) responses
      const result = await prsm.models.infer(queryOptions);

      // Log successful request
      console.log(`[${requestId}] Request completed`, {
        model: result.model,
        tokensUsed: result.usage.totalTokens,
        cost: result.cost
      });

      // Return the response
      res.status(200).json({
        content: result.content,
        model: result.model,
        usage: {
          promptTokens: result.usage.promptTokens,
          completionTokens: result.usage.completionTokens,
          totalTokens: result.usage.totalTokens
        },
        cost: result.cost,
        requestId
      });
    }

  } catch (error) {
    console.error('PRSM API Error:', error);

    // Handle different types of errors
    if (error.name === 'PRSMBudgetExceededError') {
      return res.status(402).json({
        error: 'Budget exceeded',
        message: 'Insufficient credits to complete request',
        remainingBudget: error.remainingBudget
      });
    }

    if (error.name === 'PRSMRateLimitError') {
      return res.status(429).json({
        error: 'API rate limit exceeded',
        message: 'Too many requests to PRSM API',
        retryAfter: error.retryAfter
      });
    }

    if (error.name === 'PRSMAuthenticationError') {
      return res.status(401).json({
        error: 'Authentication failed',
        message: 'Invalid API key or authentication credentials'
      });
    }

    // Generic error response
    return res.status(500).json({
      error: 'Internal server error',
      message: 'An unexpected error occurred while processing your request'
    });
  }
}

// Export configuration for Next.js
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
    responseLimit: '8mb',
  },
}