import type { NextApiRequest, NextApiResponse } from 'next';
import path from 'path';
import { LLMRouter, LLMProvider } from '../../lib/llm-clients/llm-router';
import { ConciergeEngine } from '../../lib/prompt-engine/concierge-engine';

interface ChatRequest {
  message: string;
  conversationHistory?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

interface ChatResponse {
  content: string;
  sourceReferences: string[];
  confidence: 'high' | 'medium' | 'low';
  escalationSuggested: boolean;
  responseMetadata: {
    provider: string;
    model: string;
    responseTime: number;
    tokensUsed: number;
  };
}

interface ErrorResponse {
  error: string;
  details?: string;
}

// Initialize the concierge engine (singleton pattern)
let conciergeEngine: ConciergeEngine | null = null;

async function initializeConciergeEngine(): Promise<ConciergeEngine> {
  if (conciergeEngine) {
    return conciergeEngine;
  }

  // Configure LLM router with available API keys
  const llmConfig = {
    provider: (process.env.DEFAULT_LLM_PROVIDER as LLMProvider) || 'claude',
    fallback: (['gemini', 'claude'] as LLMProvider[]).filter(p => p !== process.env.DEFAULT_LLM_PROVIDER)
  };

  const apiKeys = {
    claude: process.env.ANTHROPIC_API_KEY,
    gemini: process.env.GOOGLE_API_KEY,
    openai: process.env.OPENAI_API_KEY
  };

  // Validate that at least one API key is available
  const availableKeys = Object.entries(apiKeys).filter(([_, key]) => !!key);
  if (availableKeys.length === 0) {
    throw new Error('No LLM API keys configured. Please set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY.');
  }

  const llmRouter = new LLMRouter(llmConfig, apiKeys);
  conciergeEngine = new ConciergeEngine(llmRouter);

  // Load knowledge base (using compiled version for size optimization)
  // Try multiple path resolutions for different deployment environments
  const possiblePaths = [
    path.resolve(process.cwd(), 'knowledge-base', 'compiled-knowledge.json'),
    path.resolve(process.cwd(), 'ai-concierge', 'knowledge-base', 'compiled-knowledge.json'),
    path.resolve(__dirname, '..', '..', 'knowledge-base', 'compiled-knowledge.json'),
    path.resolve(__dirname, '..', '..', '..', 'knowledge-base', 'compiled-knowledge.json')
  ];

  console.log('Current working directory:', process.cwd());
  console.log('__dirname:', __dirname);
  console.log('Trying knowledge base paths:', possiblePaths);

  let knowledgeBasePath = null;
  for (const tryPath of possiblePaths) {
    try {
      // Check if file exists
      const fs = require('fs');
      if (fs.existsSync(tryPath)) {
        knowledgeBasePath = tryPath;
        console.log('Found knowledge base at:', tryPath);
        break;
      } else {
        console.log('Knowledge base not found at:', tryPath);
      }
    } catch (error) {
      console.log('Error checking path:', tryPath, error);
    }
  }

  if (!knowledgeBasePath) {
    throw new Error('Knowledge base file not found in any expected location');
  }

  await conciergeEngine.loadKnowledgeBase(knowledgeBasePath);

  // Test connections
  const connectionTests = await llmRouter.testConnections();
  console.log('LLM Connection Tests:', connectionTests);

  return conciergeEngine;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | ErrorResponse>
) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Add detailed logging for debugging
    console.log('Chat API called with method:', req.method);
    console.log('Environment:', process.env.NODE_ENV);
    console.log('Available env vars:', {
      hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
      hasGoogleKey: !!process.env.GOOGLE_API_KEY,
      hasOpenAIKey: !!process.env.OPENAI_API_KEY,
      defaultProvider: process.env.DEFAULT_LLM_PROVIDER
    });

    // Validate request body
    const { message, conversationHistory }: ChatRequest = req.body;
    
    if (!message || typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({ 
        error: 'Invalid request',
        details: 'Message is required and must be a non-empty string' 
      });
    }

    if (message.length > 2000) {
      return res.status(400).json({ 
        error: 'Message too long',
        details: 'Message must be 2000 characters or less' 
      });
    }

    console.log('Processing message:', message.substring(0, 100) + '...');

    // Initialize concierge engine
    const engine = await initializeConciergeEngine();
    console.log('Concierge engine initialized successfully');

    // Process the query
    const response = await engine.processInvestorQuery(message, {
      includeHistory: !!conversationHistory,
      maxContextDocs: 10
    });

    console.log('Query processed successfully');
    // Return successful response
    res.status(200).json(response);

  } catch (error) {
    console.error('Chat API Error (full details):', error);
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack available');
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Handle different types of errors
    if (errorMessage.includes('Knowledge base not loaded')) {
      return res.status(503).json({ 
        error: 'Service temporarily unavailable',
        details: 'Knowledge base is being updated. Please try again in a moment.' 
      });
    }
    
    if (errorMessage.includes('API key') || errorMessage.includes('authentication')) {
      return res.status(503).json({ 
        error: 'Service configuration error',
        details: 'AI service is temporarily unavailable. Please contact support.' 
      });
    }
    
    if (errorMessage.includes('rate limit') || errorMessage.includes('quota')) {
      return res.status(429).json({ 
        error: 'Service temporarily busy',
        details: 'Please wait a moment and try again.' 
      });
    }

    // Generic error response with more details in development
    return res.status(500).json({ 
      error: 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? 
        `${errorMessage}\n\nStack: ${error instanceof Error ? error.stack : 'No stack'}` : 
        'An unexpected error occurred. Please try again.'
    });
  }
}

// Configure API route
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
}