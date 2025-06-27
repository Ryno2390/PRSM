import type { NextApiRequest, NextApiResponse } from 'next';
import { LLMRouter, LLMProvider } from '../../lib/llm-clients/llm-router';

interface TestResponse {
  success: boolean;
  provider?: string;
  content?: string;
  error?: string;
  availableProviders?: string[];
  connectionTests?: any;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<TestResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method not allowed' });
  }

  try {
    console.log('Testing LLM router directly...');
    
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

    console.log('API Keys available:', {
      claude: !!apiKeys.claude,
      gemini: !!apiKeys.gemini,
      openai: !!apiKeys.openai
    });

    const llmRouter = new LLMRouter(llmConfig, apiKeys);
    
    // Test connections first
    const connectionTests = await llmRouter.testConnections();
    console.log('Connection tests:', connectionTests);
    
    const availableProviders = llmRouter.getAvailableProviders();
    console.log('Available providers:', availableProviders);

    // Simple test message
    const messages = [
      { role: 'user' as const, content: 'Hello, please respond with "LLM test successful"' }
    ];
    
    const systemPrompt = 'You are a helpful assistant. Respond exactly as requested.';
    
    console.log('Attempting to generate response...');
    const response = await llmRouter.generateResponse(messages, systemPrompt);
    console.log('Response received:', response);

    res.status(200).json({
      success: true,
      provider: response.provider,
      content: response.content,
      availableProviders,
      connectionTests
    });

  } catch (error) {
    console.error('LLM test error:', error);
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack');
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
}