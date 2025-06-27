import type { NextApiRequest, NextApiResponse } from 'next';

interface TestResponse {
  success: boolean;
  message: string;
  timestamp: string;
  env: string;
}

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<TestResponse>
) {
  console.log('Minimal test endpoint called');
  
  res.status(200).json({
    success: true,
    message: 'Minimal test successful - API routes are working',
    timestamp: new Date().toISOString(),
    env: process.env.NODE_ENV || 'unknown'
  });
}