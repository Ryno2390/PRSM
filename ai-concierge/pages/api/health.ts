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
      const knowledgeBasePath = path.resolve(process.cwd(), 'knowledge-base', 'compiled-knowledge.json');
      
      if (fs.existsSync(knowledgeBasePath)) {
        const stats = fs.statSync(knowledgeBasePath);
        knowledgeBase = true;
        knowledgeBaseSize = stats.size;
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