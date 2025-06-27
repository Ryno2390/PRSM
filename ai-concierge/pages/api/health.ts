import type { NextApiRequest, NextApiResponse } from 'next';

interface HealthResponse {
  status: 'ok' | 'error';
  timestamp: string;
  environment: string;
  checks: {
    knowledgeBase: boolean;
    apiKeys: {
      anthropic: boolean;
      google: boolean;
      openai: boolean;
    };
    knowledgeBaseSize?: number;
  };
  version: string;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<HealthResponse>
) {
  const timestamp = new Date().toISOString();
  
  try {
    // Check API keys
    const apiKeys = {
      anthropic: !!process.env.ANTHROPIC_API_KEY,
      google: !!process.env.GOOGLE_API_KEY,
      openai: !!process.env.OPENAI_API_KEY
    };

    // Check knowledge base
    let knowledgeBase = false;
    let knowledgeBaseSize = 0;
    
    try {
      const fs = require('fs');
      const path = require('path');
      
      const possiblePaths = [
        path.resolve(process.cwd(), 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(process.cwd(), 'ai-concierge', 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(__dirname, '..', '..', 'knowledge-base', 'compiled-knowledge.json'),
        path.resolve(__dirname, '..', '..', '..', 'knowledge-base', 'compiled-knowledge.json')
      ];

      console.log('Health check - Current working directory:', process.cwd());
      console.log('Health check - Trying knowledge base paths:', possiblePaths);
      
      for (const tryPath of possiblePaths) {
        if (fs.existsSync(tryPath)) {
          const stats = fs.statSync(tryPath);
          knowledgeBase = true;
          knowledgeBaseSize = stats.size;
          console.log('Health check - Found knowledge base at:', tryPath, 'Size:', stats.size);
          break;
        } else {
          console.log('Health check - Knowledge base not found at:', tryPath);
        }
      }
    } catch (err) {
      console.log('Knowledge base check failed:', err);
    }

    const hasAnyApiKey = Object.values(apiKeys).some(Boolean);
    const status = knowledgeBase && hasAnyApiKey ? 'ok' : 'error';

    res.status(200).json({
      status,
      timestamp,
      environment: process.env.NODE_ENV || 'unknown',
      checks: {
        knowledgeBase,
        apiKeys,
        knowledgeBaseSize
      },
      version: '1.0.0'
    });

  } catch (error) {
    console.error('Health check error:', error);
    
    res.status(500).json({
      status: 'error',
      timestamp,
      environment: process.env.NODE_ENV || 'unknown',
      checks: {
        knowledgeBase: false,
        apiKeys: {
          anthropic: false,
          google: false,
          openai: false
        }
      },
      version: '1.0.0'
    });
  }
}