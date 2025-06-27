// PRSM Tool Execution Example
// Demonstrates AI agent tool usage and execution

const PRSM = require('../index.js');

async function toolExecutionExample() {
    console.log('🚀 PRSM JavaScript SDK - Tool Execution');
    
    const client = new PRSM.Client({
        apiKey: process.env.PRSM_API_KEY || 'demo-key',
        endpoint: process.env.PRSM_ENDPOINT || 'http://localhost:8000'
    });
    
    try {
        // Define available tools
        const tools = [
            {
                name: 'calculate',
                description: 'Perform mathematical calculations',
                parameters: {
                    type: 'object',
                    properties: {
                        expression: { type: 'string', description: 'Mathematical expression to evaluate' }
                    }
                }
            },
            {
                name: 'search_docs',
                description: 'Search PRSM documentation',
                parameters: {
                    type: 'object',
                    properties: {
                        query: { type: 'string', description: 'Search query for documentation' }
                    }
                }
            }
        ];
        
        // Query with tool usage
        const response = await client.queryWithTools({
            prompt: "What is 15% of 2,500 and where can I find more information about PRSM tokenomics?",
            tools: tools,
            maxTokens: 200
        });
        
        console.log('📝 AI Response:', response.text);
        
        // Handle tool calls
        if (response.toolCalls) {
            console.log('🔧 Tool calls made:', response.toolCalls.length);
            
            for (const toolCall of response.toolCalls) {
                console.log(`  - Tool: ${toolCall.name}`);
                console.log(`  - Args: ${JSON.stringify(toolCall.arguments)}`);
                
                // Execute tool (mock implementation)
                let result;
                if (toolCall.name === 'calculate') {
                    result = eval(toolCall.arguments.expression); // Note: eval is dangerous in production
                } else if (toolCall.name === 'search_docs') {
                    result = 'Found tokenomics documentation in docs/tokenomics.md';
                }
                
                console.log(`  - Result: ${result}`);
            }
        }
        
    } catch (error) {
        console.error('❌ Tool execution error:', error.message);
    }
}

// Run the example
if (require.main === module) {
    toolExecutionExample().catch(console.error);
}

module.exports = { toolExecutionExample };