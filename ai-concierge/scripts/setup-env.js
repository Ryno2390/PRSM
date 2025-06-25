#!/usr/bin/env node

/**
 * PRSM AI Concierge - Environment Setup Script
 * 
 * This script helps configure environment variables from API keys
 * while keeping sensitive data out of Git.
 */

const fs = require('fs');
const path = require('path');

const ENV_FILE = path.resolve(__dirname, '../.env');
const API_KEYS_FILE = path.resolve(__dirname, '../API_Keys.txt');

function setupEnvironment() {
  console.log('🔧 Setting up PRSM AI Concierge environment...\n');

  // Check if API_Keys.txt exists
  if (!fs.existsSync(API_KEYS_FILE)) {
    console.log('⚠️  API_Keys.txt not found. Please create this file with your API keys.');
    console.log('Expected format:');
    console.log('ANTHROPIC_API_KEY=your_claude_key_here');
    console.log('GOOGLE_API_KEY=your_gemini_key_here');
    console.log('OPENAI_API_KEY=your_openai_key_here');
    console.log('\nExample content:');
    console.log('ANTHROPIC_API_KEY=sk-ant-api03-...');
    console.log('GOOGLE_API_KEY=AIzaSy...');
    console.log('OPENAI_API_KEY=sk-...');
    return;
  }

  try {
    // Read API keys file
    const apiKeysContent = fs.readFileSync(API_KEYS_FILE, 'utf-8');
    console.log('✅ Found API_Keys.txt file');

    // Parse key-value pairs (supports both "KEY=value" and "Provider: key" formats)
    const envVars = {};
    const lines = apiKeysContent.split('\n').filter(line => line.trim());
    
    lines.forEach(line => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith('#')) {
        // Handle "Provider: key" format
        if (trimmedLine.includes(':')) {
          const [provider, key] = trimmedLine.split(':').map(s => s.trim());
          if (provider && key) {
            const providerLower = provider.toLowerCase();
            if (providerLower === 'anthropic') {
              envVars['ANTHROPIC_API_KEY'] = key;
            } else if (providerLower === 'gemini') {
              envVars['GOOGLE_API_KEY'] = key;
            } else if (providerLower === 'openai') {
              envVars['OPENAI_API_KEY'] = key;
            } else if (providerLower === 'openrouter') {
              envVars['OPENROUTER_API_KEY'] = key;
            }
          }
        }
        // Handle "KEY=value" format  
        else if (trimmedLine.includes('=')) {
          const [key, ...valueParts] = trimmedLine.split('=');
          if (key && valueParts.length > 0) {
            envVars[key.trim()] = valueParts.join('=').trim();
          }
        }
      }
    });

    // Build .env content
    let envContent = `# PRSM AI Concierge Environment Configuration
# Generated automatically from API_Keys.txt
# DO NOT commit this file to Git

# Application Configuration
NEXT_PUBLIC_APP_URL=http://localhost:3000
DEFAULT_LLM_PROVIDER=claude
NODE_ENV=development

# LLM API Keys
`;

    // Add API keys
    const keyMapping = {
      'ANTHROPIC_API_KEY': 'Claude API Key',
      'GOOGLE_API_KEY': 'Gemini API Key', 
      'OPENAI_API_KEY': 'OpenAI API Key'
    };

    let keysFound = 0;
    for (const [envKey, description] of Object.entries(keyMapping)) {
      if (envVars[envKey]) {
        envContent += `${envKey}=${envVars[envKey]}\n`;
        console.log(`✅ Configured ${description}`);
        keysFound++;
      } else {
        envContent += `# ${envKey}=your_key_here\n`;
        console.log(`⚠️  Missing ${description}`);
      }
    }

    // Write .env file
    fs.writeFileSync(ENV_FILE, envContent);
    console.log(`\n✅ Created .env file with ${keysFound} API keys configured`);

    if (keysFound === 0) {
      console.log('\n⚠️  No valid API keys found in API_Keys.txt');
      console.log('The system will run in simulation mode.');
    } else {
      console.log('\n🚀 Ready to test with real LLM providers!');
      console.log('\nNext steps:');
      console.log('1. npm run knowledge-compile  # Ensure knowledge base is up to date');
      console.log('2. npm run prompt-test       # Test with real API calls');
      console.log('3. npm run dev              # Start development server');
    }

  } catch (error) {
    console.error('❌ Error setting up environment:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  setupEnvironment();
}

module.exports = { setupEnvironment };