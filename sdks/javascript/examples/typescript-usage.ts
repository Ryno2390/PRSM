// TypeScript PRSM SDK Usage Example
// Demonstrates type-safe PRSM operations

import { Client, QueryRequest, QueryResponse } from '../types';

interface PRSMConfig {
    apiKey: string;
    endpoint: string;
    timeout?: number;
}

async function typescriptExample(): Promise<void> {
    console.log('🚀 PRSM TypeScript SDK - Type-Safe Usage');
    
    // Initialize with strong typing
    const config: PRSMConfig = {
        apiKey: process.env.PRSM_API_KEY || 'demo-key',
        endpoint: process.env.PRSM_ENDPOINT || 'http://localhost:8000',
        timeout: 30000
    };
    
    const client = new Client(config);
    
    try {
        // Type-safe query
        const request: QueryRequest = {
            prompt: "Explain PRSM's distributed AI architecture",
            maxTokens: 200,
            temperature: 0.7,
            model: "gpt-4"
        };
        
        const response: QueryResponse = await client.query(request);
        
        console.log('📝 Response:', response.text);
        console.log('💰 Cost:', response.cost);
        console.log('🕒 Latency:', response.latency_ms);
        
    } catch (error) {
        console.error('❌ Error:', (error as Error).message);
    }
}

// Export for module usage
export { typescriptExample, PRSMConfig };

// Run if called directly
if (require.main === module) {
    typescriptExample().catch(console.error);
}