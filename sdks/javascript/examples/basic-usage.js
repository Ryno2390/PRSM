// Basic PRSM JavaScript SDK Usage Example
// This example demonstrates fundamental PRSM operations

const PRSM = require('../index.js');

async function basicExample() {
    console.log('🚀 PRSM JavaScript SDK - Basic Usage');
    
    // Initialize PRSM client
    const client = new PRSM.Client({
        apiKey: process.env.PRSM_API_KEY || 'demo-key',
        endpoint: process.env.PRSM_ENDPOINT || 'http://localhost:8000'
    });
    
    try {
        // Basic query example
        const response = await client.query({
            prompt: "Hello PRSM! What can you tell me about distributed AI?",
            maxTokens: 150
        });
        
        console.log('📝 Response:', response.text);
        console.log('💰 Cost:', response.cost);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

// Run the example
if (require.main === module) {
    basicExample().catch(console.error);
}

module.exports = { basicExample };