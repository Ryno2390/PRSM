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

// Initialize the concierge engine (singleton pattern with Netlify optimizations)
let conciergeEngine: ConciergeEngine | null = null;
let knowledgeBaseCache: any = null;
let initPromise: Promise<ConciergeEngine> | null = null;

async function initializeConciergeEngine(): Promise<ConciergeEngine> {
  // If already initializing, wait for that to complete
  if (initPromise) {
    return initPromise;
  }

  // If already initialized, return immediately
  if (conciergeEngine) {
    return conciergeEngine;
  }

  console.log('Initializing concierge engine with Netlify optimizations...');
  const startTime = Date.now();

  initPromise = (async () => {
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

    // Fast knowledge base loading with caching
    if (!knowledgeBaseCache) {
      const possiblePaths = [
        path.resolve(process.cwd(), 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(process.cwd(), 'ai-concierge', 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(__dirname, '..', '..', 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(__dirname, '..', '..', '..', 'knowledge-base', 'compiled-knowledge.json')
      ];

      console.log('Loading knowledge base...');
      let knowledgeBasePath = null;
      for (const tryPath of possiblePaths) {
        try {
          const fs = require('fs');
          if (fs.existsSync(tryPath)) {
            knowledgeBasePath = tryPath;
            console.log('Found knowledge base at:', tryPath);
            break;
          }
        } catch (error) {
          // Continue to next path
        }
      }

      if (!knowledgeBasePath) {
        throw new Error('Knowledge base file not found in any expected location');
      }

      await conciergeEngine.loadKnowledgeBase(knowledgeBasePath);
      knowledgeBaseCache = true; // Mark as loaded
    }

    const elapsed = Date.now() - startTime;
    console.log(`Concierge engine initialized in ${elapsed}ms`);

    return conciergeEngine!;
  })();

  const result = await initPromise;
  initPromise = null; // Reset for next time
  return result;
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

    // Process the query with Netlify optimizations
    console.log('Starting query processing...');
    const queryStartTime = Date.now();
    
    // NETLIFY OPTIMIZATION: Reduced context and timeout handling
    const response = await Promise.race([
      engine.processInvestorQuery(message, {
        includeHistory: false, // Disable history for speed
        maxContextDocs: 2 // Reduce to 2 docs for Netlify
      }),
      new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error('Query processing timeout after 8 seconds')), 8000)
      )
    ]);
    
    const queryElapsed = Date.now() - queryStartTime;
    console.log(`Query processed in ${queryElapsed}ms`);

    console.log('Query processed successfully');
    // Return successful response
    res.status(200).json(response);

  } catch (error) {
    console.error('Chat API Error (full details):', error);
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack available');
    console.error('Error name:', error instanceof Error ? error.name : 'Unknown');
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('Processed error message:', errorMessage);
    
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