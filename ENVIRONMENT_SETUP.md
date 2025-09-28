# Environment Setup Guide

## Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# OpenAI API Key (required for LLM and Weaviate vectorization)
OPENAI_API_KEY=your_openai_api_key_here

# Weaviate Cloud Configuration
WEAVIATE_URL=your_weaviate_url_here
WEAVIATE_API_KEY=your_weaviate_api_key_here

# LightPanda Browser Token (optional)
LIGHTPANDA_TOKEN=your_lightpanda_token_here

# LLM Configuration
OPENAI_MODEL=gpt-4o-mini
FORCE_PROVIDER=openai
```

## Setup Steps

1. **Create `.env` file**: Copy the template above and fill in your actual values
2. **Get OpenAI API Key**: Sign up at https://platform.openai.com/api-keys
3. **Get Weaviate Cloud**: Sign up at https://console.weaviate.cloud/
4. **Get LightPanda Token**: Sign up at https://lightpanda.io/ (optional)

## Testing the Setup

Run the test script to verify everything is working:

```bash
python test_weaviate_integration.py
```

## Running the Agent

Once environment variables are set, you can run the agent:

```bash
python t_agent.py "Your question here"
```

## Features

- ✅ **Weaviate Integration**: Task similarity and pattern reuse
- ✅ **LLM Caching**: Reduces API costs with intelligent caching
- ✅ **Browser Automation**: Wikipedia research automation
- ✅ **Pattern Learning**: Gets smarter over time by learning from successful patterns
