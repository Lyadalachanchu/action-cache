# Setup Instructions

## 1. Set up Environment Variables

You need to set your OpenAI API key. Choose one of these methods:

### Method A: PowerShell (Temporary)
```powershell
$env:OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

### Method B: Create .env file (Recommended)
Create a file named `.env` in the project root with:
```
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
LIGHTPANDA_TOKEN=your_lightpanda_token_here
```

### Method C: System Environment Variables (Permanent)
1. Open System Properties → Environment Variables
2. Add new variable: `OPENAI_API_KEY` = `sk-your-actual-openai-api-key-here`

## 2. Get Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

## 3. Run Setup

After setting the environment variable:
```bash
python setup_weaviate.py
```

## 4. Test the System

```bash
python test_memory.py
python enhanced_main.py
```

