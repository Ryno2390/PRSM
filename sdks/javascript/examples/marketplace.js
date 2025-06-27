// PRSM Marketplace Integration Example
// Demonstrates FTNS token operations and marketplace features

const PRSM = require('../index.js');

async function marketplaceExample() {
    console.log('🚀 PRSM JavaScript SDK - Marketplace Integration');
    
    const client = new PRSM.Client({
        apiKey: process.env.PRSM_API_KEY || 'demo-key',
        endpoint: process.env.PRSM_ENDPOINT || 'http://localhost:8000'
    });
    
    try {
        // Check FTNS balance
        const balance = await client.marketplace.getBalance();
        console.log('💰 FTNS Balance:', balance.ftns, 'tokens');
        
        // List available models in marketplace
        const models = await client.marketplace.listModels();
        console.log('🤖 Available models:', models.length);
        
        models.slice(0, 3).forEach(model => {
            console.log(`  - ${model.name}: ${model.price_per_token} FTNS/token`);
        });
        
        // Purchase model access with FTNS
        if (balance.ftns > 10) {
            const purchase = await client.marketplace.purchaseAccess({
                modelId: models[0].id,
                duration: '1h',
                ftnsAmount: 5
            });
            
            console.log('✅ Purchased access:', purchase.accessToken);
        }
        
        // Query using marketplace model
        const response = await client.query({
            prompt: "Explain PRSM's marketplace economy",
            model: models[0].id,
            maxTokens: 100
        });
        
        console.log('📝 Response:', response.text);
        console.log('💸 FTNS spent:', response.ftnsSpent);
        
    } catch (error) {
        console.error('❌ Marketplace error:', error.message);
    }
}

// Run the example
if (require.main === module) {
    marketplaceExample().catch(console.error);
}

module.exports = { marketplaceExample };