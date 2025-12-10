# DOCX Form Filler - OpenRouter + QWEN 8B

Automated form filling for DOCX documents using OpenRouter's QWEN model.

## Project Structure

```
project/
├── config.py              # Configuration and settings
├── docx_agent.py          # Main agent implementation
├── run.py                 # Execution script
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (create this)
├── sample_forms.docx      # Input template
├── input_date.json        # Input data
└── sample_forms.filled.docx  # Output (generated)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Set API Key

Create `.env` file:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
SITE_URL=https://yoursite.com
SITE_NAME=DOCX Form Filler
```

Or export directly:

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Run

```bash
python run.py
```

## Configuration

Edit `config.py` to customize:

- **Model**: `qwen/qwen-2.5-72b-instruct` (default)
- **Temperature**: `0.0` (deterministic)
- **Max Tokens**: `8192`
- **Confidence Threshold**: `0.7`

Available QWEN models on OpenRouter:
- `qwen/qwen-2.5-72b-instruct`
- `qwen/qwen-2-72b-instruct`
- `qwen/qwen-2.5-7b-instruct`

## How It Works

1. **Load Data**: Reads JSON with field values
2. **Detect Placeholders**: Finds `_____` and `....` patterns
3. **Map with QWEN**: Uses AI to match placeholders to fields
4. **Fuzzy Fallback**: Secondary matching for unmapped fields
5. **Fill Document**: Replaces placeholders while preserving formatting
6. **Save Output**: Creates `.filled.docx` file

## OpenRouter Integration

The agent uses OpenAI SDK format:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

response = client.chat.completions.create(
    model="qwen/qwen-2.5-72b-instruct",
    messages=[...],
    temperature=0.0,
    extra_headers={
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
)
```

## Input Data Format

`input_date.json`:

```json
{
  "Nume ofertant": "SC Example SRL",
  "Data": "2025-10-08",
  "CIF": "RO12345678",
  "Adresa": "Str. Test 123, Bucuresti"
}
```

## Features

✅ **Preserves formatting** - Keeps fonts, spacing, tables intact  
✅ **Multi-occurrence** - Fills all instances of same placeholder  
✅ **Unicode support** - Romanian diacritics handled correctly  
✅ **Deterministic** - Same input = same output  
✅ **Fuzzy fallback** - Works even when AI mapping fails  

## Troubleshooting

### API Key Error
```
❌ ERROR: OPENROUTER_API_KEY not set!
```
**Solution**: Set the environment variable or create `.env` file

### Model Not Found
```
Error: Model 'qwen/...' not found
```
**Solution**: Check available models at [openrouter.ai/models](https://openrouter.ai/models)

### Token Limit Exceeded
**Solution**: Reduce `MAX_PLACEHOLDERS_PER_REQUEST` in `config.py`

## Cost Estimation

QWEN 2.5 72B:
- Input: $0.36/1M tokens
- Output: $0.72/1M tokens

Typical run: ~5,000 tokens ≈ $0.004 per document

## License

MIT License - See LICENSE file

## Support

For issues with:
- OpenRouter API: [openrouter.ai/docs](https://openrouter.ai/docs)
- QWEN models: [qwen.readthedocs.io](https://qwen.readthedocs.io/)
- This project: Open an issue on GitHub
