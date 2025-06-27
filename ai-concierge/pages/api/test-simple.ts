import type { NextApiRequest, NextApiResponse } from 'next';
import { LLMRouter, LLMProvider } from '../../lib/llm-clients/llm-router';

interface TestResponse {
  success: boolean;
  content?: string;
  error?: string;
  timing?: number;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<TestResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method not allowed' });
  }

  const startTime = Date.now();

  try {
    console.log('Testing simple LLM call...');
    
    // Configure LLM router with available API keys
    const llmConfig = {
      provider: (process.env.DEFAULT_LLM_PROVIDER as LLMProvider) || 'claude',
      fallback: ['gemini', 'claude'] as LLMProvider[]
    };

    const apiKeys = {
      claude: process.env.ANTHROPIC_API_KEY,
      gemini: process.env.GOOGLE_API_KEY,
      openai: process.env.OPENAI_API_KEY
    };

    console.log('Creating LLM router...');
    const llmRouter = new LLMRouter(llmConfig, apiKeys);
    
    // Very simple test - no knowledge base, minimal prompt
    const messages = [
      { role: 'user' as const, content: 'Say "Test successful"' }
    ];
    
    const systemPrompt = 'You are a helpful assistant. Respond exactly as requested.';
    
    console.log('Calling LLM...');
    const response = await llmRouter.generateResponse(messages, systemPrompt);
    console.log('LLM response received:', response.content.substring(0, 100));

    const timing = Date.now() - startTime;

    res.status(200).json({
      success: true,
      content: response.content,
      timing
    });

  } catch (error) {
    console.error('Simple test error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    const timing = Date.now() - startTime;
    
    res.status(500).json({
      success: false,
      error: errorMessage,
      timing
    });
  }
}