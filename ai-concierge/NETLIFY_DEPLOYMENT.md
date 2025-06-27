# PRSM AI Concierge - Netlify Deployment Guide

## Current Issue: 502 Error Fix

The 502 error when asking "Tell me about PRSM" is caused by missing environment variables on Netlify. Here's how to fix it:

## Step 1: Configure Environment Variables on Netlify

1. Go to your Netlify dashboard: https://app.netlify.com/
2. Select your `prsm-concierge` site
3. Go to **Site settings** > **Environment variables**
4. Add the following environment variables:

### Required Environment Variables:

```
ANTHROPIC_API_KEY=[Copy from ai-concierge/API_Keys.txt - Anthropic key]

GOOGLE_API_KEY=[Copy from ai-concierge/API_Keys.txt - Gemini key]

OPENAI_API_KEY=[Copy from ai-concierge/API_Keys.txt - OpenAI key]

DEFAULT_LLM_PROVIDER=claude

NEXT_PUBLIC_APP_URL=https://prsm-concierge.netlify.app

NODE_ENV=production
```

**Note**: The actual API keys are stored in `ai-concierge/API_Keys.txt` in your local repository. Copy the values from that file into Netlify's environment variables.

## Step 2: Redeploy the Site

After adding the environment variables:

1. Go to **Deploys** tab in your Netlify dashboard
2. Click **Trigger deploy** > **Deploy site**
3. Wait for the deployment to complete (should take 2-3 minutes)

## Step 3: Test the Fix

1. Visit https://prsm-concierge.netlify.app
2. Ask the question: "Tell me about PRSM"
3. The response should now work without the 502 error

## Step 4: Health Check

You can also test the health endpoint to verify configuration:
- Visit: https://prsm-concierge.netlify.app/api/health
- This will show the status of API keys and knowledge base

## Expected Health Check Response:

```json
{
  "status": "ok",
  "timestamp": "2025-06-27T...",
  "environment": "production",
  "checks": {
    "knowledgeBase": true,
    "apiKeys": {
      "anthropic": true,
      "google": true,
      "openai": true
    },
    "knowledgeBaseSize": 436824
  },
  "version": "1.0.0"
}
```

## Troubleshooting

If you still get errors:

1. **Check Environment Variables**: Ensure all API keys are correctly set in Netlify
2. **Check Build Logs**: Look at the deployment logs for any build errors
3. **Check Function Logs**: Enable function logs in Netlify to see runtime errors
4. **Test Locally**: Run `npm run dev` locally to ensure the code works

## Recent Fixes Applied

1. **Enhanced error logging** in `/pages/api/chat.ts`
2. **Added health check endpoint** at `/api/health`
3. **Updated Netlify configuration** in `netlify.toml`
4. **Improved build process** with `build-with-knowledge` command

The 502 error should be resolved once the environment variables are properly configured on Netlify.