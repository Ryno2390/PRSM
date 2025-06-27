// PRSM Streaming Response Example
// Demonstrates real-time streaming capabilities

const PRSM = require('../index.js');

async function streamingExample() {
    console.log('🚀 PRSM JavaScript SDK - Streaming Responses');
    
    const client = new PRSM.Client({
        apiKey: process.env.PRSM_API_KEY || 'demo-key',
        endpoint: process.env.PRSM_ENDPOINT || 'http://localhost:8000'
    });
    
    try {
        console.log('📡 Starting streaming request...');
        
        // Stream a longer response
        const stream = await client.streamQuery({
            prompt: "Write a detailed explanation of PRSM's tokenomics and how FTNS tokens work",
            maxTokens: 500,
            stream: true
        });
        
        process.stdout.write('📝 Streaming response: ');
        
        // Handle streaming data
        for await (const chunk of stream) {
            if (chunk.delta) {
                process.stdout.write(chunk.delta);
            }
            
            if (chunk.finished) {
                console.log('\n✅ Stream completed');
                console.log('💰 Total cost:', chunk.totalCost);
                console.log('🕒 Total time:', chunk.totalTime, 'ms');
                break;
            }
        }
        
    } catch (error) {
        console.error('❌ Streaming error:', error.message);
    }
}

// Run the example
if (require.main === module) {
    streamingExample().catch(console.error);
}

module.exports = { streamingExample };