# Environment Variables Setup

This project uses environment variables for API keys and configuration.

## Quick Setup

1. **Create a `.env` file** in the project root:
   ```bash
   # Copy the example (if it exists)
   cp .env.example .env
   
   # Or create manually
   touch .env
   ```

2. **Add your API keys** to `.env`:
   ```bash
   # Google Gemini API Key
   GEMINI_API_KEY=your_actual_api_key_here
   
   # OpenAI API Key (for future use)
   # OPENAI_API_KEY=your_openai_key_here
   
   # Anthropic API Key (for future use)
   # ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

3. **Install python-dotenv** (optional but recommended):
   ```bash
   pip install python-dotenv
   ```

## Getting API Keys

### Google Gemini
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to `.env` as `GEMINI_API_KEY`

### OpenAI (for future use)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in and create a new API key
3. Add to `.env` as `OPENAI_API_KEY`

### Anthropic (for future use)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. Add to `.env` as `ANTHROPIC_API_KEY`

## Alternative: Environment Variables

If you prefer not to use `.env` files, you can set environment variables directly:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_key_here"
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your_key_here
```

## Security Notes

- ⚠️ **Never commit `.env` files to git** - they're already in `.gitignore`
- ✅ The `.env.example` file (if it exists) is safe to commit as a template
- ✅ API keys in environment variables are also secure (not in git)

## Verification

To verify your API key is loaded:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print("✅ API key loaded successfully")
else:
    print("❌ API key not found")
```



