#!/usr/bin/env node
/**
 * PRSM JavaScript SDK Quick Start Examples
 * 
 * This file demonstrates the core capabilities of the PRSM platform
 * using the JavaScript/Node.js SDK.
 * 
 * Examples included:
 * 1. Basic agent creation and execution
 * 2. Multi-model orchestration
 * 3. Real-time streaming with WebSockets
 * 4. Browser-based AI applications
 * 5. Cost optimization
 * 6. P2P network integration
 * 7. Workflow automation
 * 8. Error handling and monitoring
 * 
 * Installation: npm install @prsm/sdk
 * Usage: node quickstart.js
 */

// Import PRSM SDK (mock implementation for demonstration)
// In practice, install with: npm install @prsm/sdk

class PRSMClient {
    constructor(options = {}) {
        this.apiKey = options.apiKey || process.env.PRSM_API_KEY || 'demo_key';
        this.baseUrl = options.baseUrl || 'https://api.prsm.network';
        this.agents = new AgentManager(this);
        this.models = new ModelManager(this);
        this.p2p = new P2PManager(this);
        this.workflows = new WorkflowManager(this);
        this.cost = new CostManager(this);
        this.websocket = new WebSocketManager(this);
    }

    async authenticate() {
        console.log('🔐 Authenticating with PRSM API...');
        console.log(`✅ Connected to ${this.baseUrl}`);
        return true;
    }
}

class AgentManager {
    constructor(client) {
        this.client = client;
    }

    async create(config) {
        return new Agent(config);
    }

    async list(options = {}) {
        return Array.from({ length: 3 }, (_, i) => 
            new Agent({ id: `agent_${i}`, name: `demo_agent_${i}` })
        );
    }
}

class ModelManager {
    constructor(client) {
        this.client = client;
    }

    getProviders() {
        return ['openai', 'anthropic', 'huggingface', 'local'];
    }

    async executeInference(options) {
        return {
            output: 'Sample inference result',
            confidence: 0.95,
            tokensUsed: 150
        };
    }
}

class P2PManager {
    constructor(client) {
        this.client = client;
    }

    async createNode(config) {
        return new P2PNode(config);
    }

    async discoverPeers() {
        return Array.from({ length: 5 }, (_, i) => ({
            id: `peer_${i}`,
            capabilities: ['inference'],
            latency: Math.random() * 100
        }));
    }
}

class WorkflowManager {
    constructor(client) {
        this.client = client;
    }

    async create(name, steps) {
        return new Workflow(name, steps);
    }
}

class CostManager {
    constructor(client) {
        this.client = client;
    }

    async getUsage() {
        return {
            totalCost: 145.67,
            tokensUsed: 125000,
            requestCount: 2500
        };
    }
}

class WebSocketManager {
    constructor(client) {
        this.client = client;
    }

    connect(endpoint) {
        // Mock WebSocket connection
        return {
            onMessage: (callback) => {
                // Simulate real-time updates
                let progress = 0;
                const interval = setInterval(() => {
                    progress += 20;
                    callback({
                        type: 'progress',
                        progress,
                        status: `Processing... ${progress}%`
                    });
                    
                    if (progress >= 100) {
                        clearInterval(interval);
                        callback({
                            type: 'complete',
                            result: 'Task completed successfully!'
                        });
                    }
                }, 500);
            },
            send: (data) => console.log('📤 Sent:', data),
            close: () => console.log('🔌 WebSocket connection closed')
        };
    }
}

class Agent {
    constructor(config) {
        this.id = config.id || `agent_${Math.floor(Math.random() * 1000)}`;
        this.name = config.name || 'demo_agent';
        this.type = config.type || 'researcher';
        this.config = config;
    }

    async execute(prompt, options = {}) {
        // Simulate execution delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        return {
            executionId: `exec_${this.id}`,
            status: 'completed',
            output: `Response to: ${prompt.substring(0, 50)}...`,
            confidence: 0.92,
            tokensUsed: 150,
            executionTime: 1.2
        };
    }

    async *stream(prompt, options = {}) {
        const stages = [
            'Initializing...',
            'Processing input...',
            'Generating response...',
            'Finalizing output...'
        ];

        for (let i = 0; i < stages.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 800));
            yield {
                progress: (i + 1) * 25,
                status: stages[i],
                partialOutput: `Partial result ${i + 1}`
            };
        }
    }
}

class P2PNode {
    constructor(config) {
        this.id = `node_${Math.floor(Math.random() * 1000)}`;
        this.config = config;
    }

    async startServices() {
        console.log(`🚀 Starting P2P services for node ${this.id}`);
        return true;
    }

    async getEarnings() {
        return {
            ftnsEarned: 45.67,
            usdEquivalent: 228.35
        };
    }
}

class Workflow {
    constructor(name, steps) {
        this.name = name;
        this.steps = steps;
        this.id = `workflow_${Math.floor(Math.random() * 1000)}`;
    }

    async execute(data) {
        return {
            workflowId: this.id,
            status: 'completed',
            results: 'Workflow completed successfully'
        };
    }
}

// Initialize PRSM client
const client = new PRSMClient();

// Example 1: Basic Agent Creation and Execution
async function example1BasicAgentCreation() {
    console.log('\n' + '='.repeat(60));
    console.log('🤖 Example 1: Basic Agent Creation and Execution');
    console.log('='.repeat(60));

    // Authenticate
    await client.authenticate();

    // Create a research agent
    const researcher = await client.agents.create({
        name: 'market_researcher',
        type: 'researcher',
        modelProvider: 'openai',
        modelName: 'gpt-4',
        capabilities: [
            'market_analysis',
            'trend_identification',
            'competitive_intelligence',
            'report_generation'
        ],
        specializedKnowledge: 'business_strategy,market_research,consumer_behavior'
    });

    console.log(`✅ Created agent: ${researcher.name} (ID: ${researcher.id})`);

    // Execute a market research task
    const researchPrompt = `
        Analyze the current market trends in sustainable technology.
        Focus on:
        1. Renewable energy adoption rates
        2. Consumer preferences for eco-friendly products
        3. Investment patterns in green technology
        4. Regulatory impacts on market growth
        
        Provide actionable insights for strategic planning.
    `;

    console.log('🔄 Executing market research task...');
    const result = await researcher.execute(researchPrompt, {
        context: {
            domain: 'sustainable_technology',
            priority: 'high',
            targetAudience: 'executives'
        }
    });

    console.log('📊 Execution completed:');
    console.log(`   Status: ${result.status}`);
    console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`   Tokens used: ${result.tokensUsed}`);
    console.log(`   Execution time: ${result.executionTime}s`);
    console.log(`   Output preview: ${result.output.substring(0, 100)}...`);

    return researcher;
}

// Example 2: Real-time Streaming with WebSockets
async function example2RealTimeStreaming() {
    console.log('\n' + '='.repeat(60));
    console.log('📡 Example 2: Real-time Streaming');
    console.log('='.repeat(60));

    // Create a streaming-capable agent
    const streamingAgent = await client.agents.create({
        name: 'real_time_analyzer',
        type: 'analyst',
        modelProvider: 'anthropic',
        modelName: 'claude-3-haiku',
        capabilities: ['real_time_analysis', 'data_streaming']
    });

    console.log(`✅ Created streaming agent: ${streamingAgent.name}`);

    const streamingPrompt = `
        Provide a comprehensive analysis of cryptocurrency market movements.
        Include real-time insights on:
        1. Price volatility patterns
        2. Trading volume analysis
        3. Market sentiment indicators
        4. Technical analysis signals
        5. Risk assessment and recommendations
    `;

    console.log('🔄 Starting streaming execution...');
    console.log('📺 Real-time updates:');

    // Stream response using async iterator
    for await (const update of streamingAgent.stream(streamingPrompt)) {
        const progressBar = '█'.repeat(Math.floor(update.progress / 5)) + 
                           '░'.repeat(20 - Math.floor(update.progress / 5));
        console.log(`   [${progressBar}] ${update.progress.toString().padStart(3)}% - ${update.status}`);
        
        if (update.partialOutput) {
            console.log(`   💬 ${update.partialOutput}`);
        }
    }

    console.log('✅ Streaming execution completed!');

    // Also demonstrate WebSocket connection
    console.log('\n🔌 Establishing WebSocket connection...');
    const ws = client.websocket.connect('wss://ws.prsm.network/stream');
    
    return new Promise((resolve) => {
        ws.onMessage((message) => {
            if (message.type === 'progress') {
                console.log(`   WebSocket update: ${message.status}`);
            } else if (message.type === 'complete') {
                console.log(`   ✅ ${message.result}`);
                ws.close();
                resolve(streamingAgent);
            }
        });

        ws.send({ action: 'start_analysis', prompt: 'Quick market update' });
    });
}

// Example 3: Multi-Model Orchestration
async function example3MultiModelOrchestration() {
    console.log('\n' + '='.repeat(60));
    console.log('🎭 Example 3: Multi-Model Orchestration');
    console.log('='.repeat(60));

    // Create specialized agents for different tasks
    const agents = {
        creative: await client.agents.create({
            name: 'creative_writer',
            modelProvider: 'openai',
            modelName: 'gpt-4',
            capabilities: ['creative_writing', 'storytelling']
        }),
        technical: await client.agents.create({
            name: 'technical_expert',
            modelProvider: 'anthropic',
            modelName: 'claude-3-sonnet',
            capabilities: ['technical_writing', 'code_review']
        }),
        analyst: await client.agents.create({
            name: 'data_analyst',
            modelProvider: 'huggingface',
            modelName: 'llama2-70b',
            capabilities: ['data_analysis', 'statistical_modeling']
        })
    };

    console.log(`✅ Created ${Object.keys(agents).length} specialized agents`);

    // Define intelligent task routing
    const taskRouter = {
        'creative_content': 'creative',
        'technical_documentation': 'technical',
        'data_analysis': 'analyst'
    };

    // Execute different types of tasks
    const tasks = [
        {
            type: 'creative_content',
            prompt: 'Write a compelling story about AI transforming scientific research'
        },
        {
            type: 'technical_documentation',
            prompt: 'Create API documentation for a machine learning service'
        },
        {
            type: 'data_analysis',
            prompt: 'Analyze customer segmentation patterns from e-commerce data'
        }
    ];

    console.log('🔄 Executing tasks with intelligent routing...');

    const results = [];
    
    for (let i = 0; i < tasks.length; i++) {
        const task = tasks[i];
        const agentKey = taskRouter[task.type];
        const agent = agents[agentKey];

        console.log(`\n📝 Task ${i + 1}: ${task.type}`);
        console.log(`   🤖 Routed to: ${agent.name}`);

        const result = await agent.execute(task.prompt);
        results.push(result);

        console.log(`   ✅ Completed with ${(result.confidence * 100).toFixed(1)}% confidence`);
        console.log(`   📊 Tokens: ${result.tokensUsed}, Time: ${result.executionTime}s`);
    }

    return { agents, results };
}

// Example 4: Browser-based AI Application
async function example4BrowserIntegration() {
    console.log('\n' + '='.repeat(60));
    console.log('🌐 Example 4: Browser Integration (Simulation)');
    console.log('='.repeat(60));

    // Simulate browser-based functionality
    const browserApp = {
        // Chat interface simulation
        chatInterface: {
            messages: [],
            
            async sendMessage(message, agentId) {
                console.log(`💬 User: ${message}`);
                
                // Find agent (in real app, this would be from state management)
                const agent = await client.agents.create({
                    name: 'chat_assistant',
                    type: 'assistant',
                    modelProvider: 'openai',
                    modelName: 'gpt-3.5-turbo'
                });

                const response = await agent.execute(message, {
                    context: { conversationType: 'interactive_chat' }
                });

                console.log(`🤖 Assistant: ${response.output}`);
                
                this.messages.push(
                    { role: 'user', content: message },
                    { role: 'assistant', content: response.output }
                );

                return response;
            }
        },

        // Real-time collaboration simulation
        collaboration: {
            participants: [],
            
            async addParticipant(name, agentConfig) {
                const agent = await client.agents.create({
                    name: `${name}_assistant`,
                    ...agentConfig
                });
                
                this.participants.push({ name, agent });
                console.log(`👤 ${name} joined the collaboration`);
                return agent;
            },

            async broadcast(message) {
                console.log(`📢 Broadcasting: ${message}`);
                
                for (const participant of this.participants) {
                    const response = await participant.agent.execute(
                        `Respond to this collaborative message: ${message}`
                    );
                    console.log(`   ${participant.name}: ${response.output.substring(0, 80)}...`);
                }
            }
        }
    };

    // Simulate chat conversation
    console.log('💬 Simulating chat interface...');
    await browserApp.chatInterface.sendMessage(
        'Help me plan a sustainable business strategy'
    );
    await browserApp.chatInterface.sendMessage(
        'What are the key environmental considerations?'
    );

    console.log(`📝 Chat history: ${browserApp.chatInterface.messages.length} messages`);

    // Simulate real-time collaboration
    console.log('\n👥 Simulating collaborative workspace...');
    
    await browserApp.collaboration.addParticipant('Designer', {
        type: 'creative',
        capabilities: ['ui_design', 'user_experience']
    });
    
    await browserApp.collaboration.addParticipant('Developer', {
        type: 'coder',
        capabilities: ['web_development', 'api_design']
    });

    await browserApp.collaboration.broadcast(
        'Let\'s design a user-friendly dashboard for environmental data'
    );

    return browserApp;
}

// Example 5: Cost Optimization and Analytics
async function example5CostOptimization() {
    console.log('\n' + '='.repeat(60));
    console.log('💰 Example 5: Cost Optimization');
    console.log('='.repeat(60));

    // Get current usage statistics
    console.log('📊 Analyzing current usage...');
    const usage = await client.cost.getUsage();
    console.log(`   Total cost this month: $${usage.totalCost.toFixed(2)}`);
    console.log(`   Tokens used: ${usage.tokensUsed.toLocaleString()}`);
    console.log(`   Requests made: ${usage.requestCount.toLocaleString()}`);

    // Cost optimization analysis
    console.log('\n🎯 Running cost optimization analysis...');

    const providers = ['openai', 'anthropic', 'huggingface', 'p2p_network'];
    const costAnalysis = {};

    for (const provider of providers) {
        // Simulated cost calculation
        const baseCosts = {
            openai: 0.002,
            anthropic: 0.0015,
            huggingface: 0.001,
            p2p_network: 0.0005
        };

        const monthlyCost = usage.tokensUsed * baseCosts[provider];
        const savingsVsCurrent = usage.totalCost - monthlyCost;
        
        costAnalysis[provider] = {
            monthlyCost,
            costPerToken: baseCosts[provider],
            savingsVsCurrent,
            savingsPercentage: (savingsVsCurrent / usage.totalCost) * 100
        };
    }

    console.log('📋 Cost analysis by provider:');
    Object.entries(costAnalysis).forEach(([provider, analysis]) => {
        const savings = analysis.savingsVsCurrent;
        const savingsPct = analysis.savingsPercentage;
        console.log(`   ${provider.padEnd(12)}: $${analysis.monthlyCost.toFixed(2).padStart(8)} (${savingsPct >= 0 ? '+' : ''}${savingsPct.toFixed(1)}%)`);
    });

    // Find best option
    const bestProvider = Object.keys(costAnalysis).reduce((best, current) => 
        costAnalysis[current].monthlyCost < costAnalysis[best].monthlyCost ? current : best
    );
    
    const bestSavings = costAnalysis[bestProvider].savingsVsCurrent;

    console.log('\n💡 Optimization recommendation:');
    console.log(`   Best provider: ${bestProvider}`);
    console.log(`   Potential savings: $${bestSavings.toFixed(2)}/month (${(bestSavings/usage.totalCost*100).toFixed(1)}%)`);

    // Real-time cost monitoring setup
    console.log('\n📈 Setting up real-time cost monitoring...');
    
    const costMonitor = {
        budget: 200.00, // Monthly budget
        currentSpend: usage.totalCost,
        alerts: [],
        
        checkBudget() {
            const percentUsed = (this.currentSpend / this.budget) * 100;
            
            if (percentUsed > 90) {
                this.alerts.push({
                    level: 'critical',
                    message: `Budget critically exceeded: ${percentUsed.toFixed(1)}% used`
                });
            } else if (percentUsed > 75) {
                this.alerts.push({
                    level: 'warning',
                    message: `Budget warning: ${percentUsed.toFixed(1)}% used`
                });
            }
            
            return this.alerts;
        }
    };

    const alerts = costMonitor.checkBudget();
    if (alerts.length > 0) {
        console.log('🚨 Budget alerts:');
        alerts.forEach(alert => {
            const icon = alert.level === 'critical' ? '🔴' : '🟡';
            console.log(`   ${icon} ${alert.message}`);
        });
    } else {
        console.log('✅ Budget status: Within limits');
    }

    return { costAnalysis, costMonitor };
}

// Example 6: P2P Network Integration
async function example6P2PIntegration() {
    console.log('\n' + '='.repeat(60));
    console.log('🌐 Example 6: P2P Network Integration');
    console.log('='.repeat(60));

    // Discover available peers
    console.log('🔍 Discovering P2P network peers...');
    const peers = await client.p2p.discoverPeers();
    console.log(`   Found ${peers.length} available peers`);

    peers.slice(0, 3).forEach(peer => {
        console.log(`   - Peer ${peer.id}: ${peer.capabilities.join(', ')} (${peer.latency.toFixed(1)}ms)`);
    });

    // Create and configure P2P node
    console.log('\n🏗️ Creating P2P node...');
    const node = await client.p2p.createNode({
        nodeType: 'hybrid_provider',
        capabilities: ['gpt-3.5-turbo', 'llama2-7b', 'stable-diffusion'],
        resources: {
            gpuMemory: '16GB',
            cpuCores: 8,
            bandwidth: '1Gbps'
        },
        pricing: {
            inferenceRate: 0.0008, // per token
            availability: '20/7', // 20 hours per day
            acceptedTokens: ['FTNS', 'USD']
        }
    });

    console.log(`✅ Created P2P node: ${node.id}`);

    // Start providing services
    await node.startServices();
    console.log('🚀 Node is now providing services to the P2P network');

    // Simulate earning tracking
    const earnings = await node.getEarnings();
    console.log('\n💰 Node earnings:');
    console.log(`   FTNS tokens earned: ${earnings.ftnsEarned}`);
    console.log(`   USD equivalent: $${earnings.usdEquivalent.toFixed(2)}`);

    // Use P2P network for distributed inference
    console.log('\n🔄 Using P2P network for distributed tasks...');

    const p2pTasks = [
        'Generate creative content for social media campaign',
        'Analyze financial risk in investment portfolio',
        'Create technical documentation for API endpoints'
    ];

    const p2pResults = [];
    
    for (let i = 0; i < p2pTasks.length; i++) {
        const task = p2pTasks[i];
        console.log(`   📤 Submitting task ${i + 1}: ${task.substring(0, 40)}...`);
        
        // In real implementation, this would route to optimal P2P peer
        const result = await client.models.executeInference({
            prompt: task,
            provider: 'p2p_network',
            routing: 'latency_optimized'
        });
        
        p2pResults.push(result);
        console.log(`   📥 Received result with ${(result.confidence * 100).toFixed(1)}% confidence`);
    }

    // Network performance metrics
    console.log('\n📊 P2P Network Performance:');
    console.log(`   Average response time: ${(Math.random() * 2 + 0.5).toFixed(1)}s`);
    console.log(`   Network uptime: 99.7%`);
    console.log(`   Active nodes: 1,247`);
    console.log(`   Daily transactions: 15,432`);

    return { node, p2pResults };
}

// Example 7: Workflow Automation
async function example7WorkflowAutomation() {
    console.log('\n' + '='.repeat(60));
    console.log('🔬 Example 7: Workflow Automation');
    console.log('='.repeat(60));

    // Define a content creation workflow
    const contentWorkflow = await client.workflows.create(
        'content_creation_pipeline',
        [
            {
                name: 'topic_research',
                agentType: 'researcher',
                input: 'topic_brief',
                output: 'research_findings'
            },
            {
                name: 'content_outline',
                agentType: 'content_strategist',
                input: 'research_findings',
                output: 'content_outline'
            },
            {
                name: 'content_writing',
                agentType: 'writer',
                input: 'content_outline',
                output: 'draft_content'
            },
            {
                name: 'content_editing',
                agentType: 'editor',
                input: 'draft_content',
                output: 'polished_content'
            },
            {
                name: 'seo_optimization',
                agentType: 'seo_specialist',
                input: 'polished_content',
                output: 'optimized_content'
            }
        ]
    );

    console.log(`✅ Created workflow: ${contentWorkflow.name}`);
    console.log(`   Pipeline steps: ${contentWorkflow.steps.length}`);

    // Execute the workflow
    const inputData = {
        topicBrief: 'Sustainable technology trends in 2025',
        targetAudience: 'Business executives and investors',
        contentType: 'Thought leadership article',
        keywords: ['sustainability', 'green technology', 'innovation', 'investment'],
        wordCount: 1500
    };

    console.log('\n🔄 Executing content creation workflow...');
    console.log(`   Topic: ${inputData.topicBrief}`);
    console.log(`   Target: ${inputData.targetAudience}`);

    const workflowResult = await contentWorkflow.execute(inputData);

    console.log('✅ Workflow execution completed:');
    console.log(`   Status: ${workflowResult.status}`);
    console.log(`   Workflow ID: ${workflowResult.workflowId}`);

    // Simulate step-by-step progress
    const stepResults = [
        { step: 'Topic Research', duration: '2.3min', output: 'Comprehensive market analysis completed' },
        { step: 'Content Outline', duration: '1.1min', output: '7-section article structure created' },
        { step: 'Content Writing', duration: '4.7min', output: '1,487-word draft article generated' },
        { step: 'Content Editing', duration: '2.1min', output: 'Grammar, style, and flow improvements applied' },
        { step: 'SEO Optimization', duration: '1.4min', output: 'Keywords integrated, meta tags optimized' }
    ];

    console.log('\n📊 Workflow Step Results:');
    stepResults.forEach((step, index) => {
        console.log(`   ${index + 1}. ${step.step} (${step.duration}): ${step.output}`);
    });

    // Workflow analytics
    const totalDuration = stepResults.reduce((sum, step) => 
        sum + parseFloat(step.duration), 0
    );

    console.log('\n📈 Workflow Performance:');
    console.log(`   Total execution time: ${totalDuration.toFixed(1)} minutes`);
    console.log(`   Steps completed: ${stepResults.length}/${contentWorkflow.steps.length}`);
    console.log(`   Success rate: 100%`);
    console.log(`   Estimated cost: $8.45`);

    return { contentWorkflow, workflowResult, stepResults };
}

// Example 8: Error Handling and Monitoring
async function example8ErrorHandling() {
    console.log('\n' + '='.repeat(60));
    console.log('🚨 Example 8: Error Handling and Monitoring');
    console.log('='.repeat(60));

    // Error handling examples
    const errorScenarios = [
        {
            name: 'Agent Creation with Invalid Config',
            test: async () => {
                try {
                    await client.agents.create({
                        name: '',  // Invalid empty name
                        modelProvider: 'invalid_provider'
                    });
                } catch (error) {
                    return { caught: true, error };
                }
                return { caught: false };
            }
        },
        {
            name: 'Execution Timeout',
            test: async () => {
                try {
                    const agent = await client.agents.create({
                        name: 'timeout_test_agent',
                        type: 'researcher'
                    });
                    
                    // Simulate timeout scenario
                    await agent.execute('Complex analysis task', {
                        timeout: 1 // Very short timeout
                    });
                } catch (error) {
                    return { caught: true, error: { message: 'Execution timeout after 1s' } };
                }
                return { caught: false };
            }
        },
        {
            name: 'Rate Limit Exceeded',
            test: async () => {
                // Simulate rate limit scenario
                return { 
                    caught: true, 
                    error: { 
                        message: 'Rate limit exceeded: 1000 requests/hour',
                        retryAfter: 3600 
                    } 
                };
            }
        }
    ];

    console.log('🧪 Testing error handling scenarios:');
    
    for (const scenario of errorScenarios) {
        console.log(`\n   Testing: ${scenario.name}`);
        const result = await scenario.test();
        
        if (result.caught) {
            console.log(`   ✅ Error properly caught: ${result.error.message}`);
        } else {
            console.log(`   ❌ Error not caught - this is unexpected`);
        }
    }

    // Monitoring and health checks
    console.log('\n📊 System Health Monitoring:');
    
    const healthMetrics = {
        apiResponseTime: Math.random() * 200 + 50, // 50-250ms
        successRate: 0.987, // 98.7%
        activeAgents: 1247,
        queuedTasks: 23,
        p2pNetworkNodes: 5439,
        lastIncident: '2024-12-15T14:30:00Z'
    };

    console.log(`   API Response Time: ${healthMetrics.apiResponseTime.toFixed(0)}ms`);
    console.log(`   Success Rate: ${(healthMetrics.successRate * 100).toFixed(1)}%`);
    console.log(`   Active Agents: ${healthMetrics.activeAgents.toLocaleString()}`);
    console.log(`   Queued Tasks: ${healthMetrics.queuedTasks}`);
    console.log(`   P2P Network Nodes: ${healthMetrics.p2pNetworkNodes.toLocaleString()}`);

    // Performance alerts
    const alerts = [];
    
    if (healthMetrics.apiResponseTime > 200) {
        alerts.push({ level: 'warning', message: 'API response time elevated' });
    }
    
    if (healthMetrics.successRate < 0.95) {
        alerts.push({ level: 'critical', message: 'Success rate below threshold' });
    }
    
    if (healthMetrics.queuedTasks > 100) {
        alerts.push({ level: 'warning', message: 'High task queue volume' });
    }

    if (alerts.length > 0) {
        console.log('\n🚨 Active Alerts:');
        alerts.forEach(alert => {
            const icon = alert.level === 'critical' ? '🔴' : '🟡';
            console.log(`   ${icon} ${alert.message}`);
        });
    } else {
        console.log('\n✅ All systems operating normally');
    }

    return { errorScenarios, healthMetrics, alerts };
}

// Main execution function
async function main() {
    console.log('🚀 PRSM JavaScript SDK - Comprehensive Examples');
    console.log('='.repeat(60));
    console.log('This demonstration showcases PRSM capabilities in JavaScript/Node.js:');
    console.log('- Agent creation and task execution');
    console.log('- Real-time streaming and WebSocket connections');
    console.log('- Multi-model orchestration and routing');
    console.log('- Browser integration patterns');
    console.log('- Cost optimization and analytics');
    console.log('- P2P network participation');
    console.log('- Workflow automation');
    console.log('- Error handling and monitoring');

    try {
        // Run all examples
        const agent = await example1BasicAgentCreation();
        const streamingAgent = await example2RealTimeStreaming();
        const { agents, results } = await example3MultiModelOrchestration();
        const browserApp = await example4BrowserIntegration();
        const { costAnalysis, costMonitor } = await example5CostOptimization();
        const { node, p2pResults } = await example6P2PIntegration();
        const { contentWorkflow, workflowResult } = await example7WorkflowAutomation();
        const { healthMetrics, alerts } = await example8ErrorHandling();

        // Summary
        console.log('\n' + '='.repeat(60));
        console.log('🎉 All Examples Completed Successfully!');
        console.log('='.repeat(60));
        console.log('Summary of what we accomplished:');
        console.log(`✅ Created ${Object.keys(agents).length + 2} AI agents`);
        console.log(`✅ Executed ${results.length} multi-model tasks`);
        console.log(`✅ Processed ${p2pResults.length} P2P network tasks`);
        console.log(`✅ Deployed P2P node: ${node.id}`);
        console.log(`✅ Completed workflow: ${contentWorkflow.name}`);
        console.log(`✅ Browser app with ${browserApp.collaboration.participants.length} participants`);
        console.log(`✅ System health: ${(healthMetrics.successRate * 100).toFixed(1)}% success rate`);

        console.log('\n🎯 Next Steps:');
        console.log('- Integrate PRSM into your web applications');
        console.log('- Explore real-time collaborative features');
        console.log('- Build custom workflows for your use cases');
        console.log('- Join the PRSM developer community');

        console.log('\n📚 Resources:');
        console.log('  📖 Docs: https://docs.prsm.network');
        console.log('  💬 Community: https://discord.gg/prsm');
        console.log('  🐙 GitHub: https://github.com/Ryno2390/PRSM');
        console.log('  📦 NPM: https://npmjs.com/package/@prsm/sdk');

    } catch (error) {
        console.error('\n❌ Error running examples:', error.message);
        console.log('Please check your API key and network connection');
        return false;
    }

    return true;
}

// Execute if running directly
if (require.main === module) {
    main().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = {
    PRSMClient,
    main,
    example1BasicAgentCreation,
    example2RealTimeStreaming,
    example3MultiModelOrchestration,
    example4BrowserIntegration,
    example5CostOptimization,
    example6P2PIntegration,
    example7WorkflowAutomation,
    example8ErrorHandling
};