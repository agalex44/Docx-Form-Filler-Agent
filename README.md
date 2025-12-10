# DOCX Form Filler - OpenRouter + QWEN

## Key Changes

- ✅ **Fixed model**: `qwen/qwen-2.5-72b-instruct` (was: embedding model)
- ✅ Enhanced context extraction (200 chars before, 50 after)
- ✅ Structured prompting with reasoning validation
- ✅ Multiple fuzzy matching algorithms
- ✅ Table cell handling
- ✅ Debug mode for troubleshooting

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key in .env
OPENROUTER_API_KEY=sk-or-v1-your-key

# 3. Run
python run.py
```

## How Mapping Works

```
Document: "Numele ofertantului: _____"
         â†"
Context: "...calitate de ofertant. Numele ofertantului: _____  ne oferim..."
         â†"
LLM analyzes: "numele" (name) + "ofertantului" (of offerant) = matches "Denumirea / numele ofertantului"
         â†"
Fills: "SC BestBuild SRL"
```

## Configuration

**Edit `config.py`:**

```python
MODEL_NAME = "qwen/qwen-2.5-72b-instruct"  # Chat model
CONFIDENCE_THRESHOLD = 0.75  # Higher = stricter
FUZZY_MATCH_THRESHOLD = 85   # Higher = stricter
DEBUG_MODE = True            # Shows detailed logs
```

**Available Models:**
- `qwen/qwen-2.5-72b-instruct` (recommended, $0.36/$0.72 per 1M)
- `qwen/qwen-2-72b-instruct`
- `qwen/qwen-2.5-7b-instruct` (cheaper, less accurate)

## Debugging

Enable debug mode in `config.py`:

```python
DEBUG_MODE = True
```

Shows:
- LLM response content
- Each mapping decision with reasoning
- Fuzzy match scores
- Unmapped placeholders with context

## Troubleshooting

### "Dates filling everywhere"
**Cause**: Embedding model can't generate text
**Fix**: Use chat model (already fixed in config)

### "Low fill rate"
**Solutions**:
1. Lower `CONFIDENCE_THRESHOLD` to 0.7
2. Lower `FUZZY_MATCH_THRESHOLD` to 80
3. Check JSON keys match document language
4. Enable `DEBUG_MODE` to see why mappings fail

### "Wrong values in wrong fields"
**Solutions**:
1. Increase `CONTEXT_CHARS_BEFORE` to 300
2. Add more examples to JSON
3. Check Romanian diacritics are correct

## Cost Estimation

QWEN 2.5 72B: ~5K tokens/doc ≈ **$0.004 per document**

## Architecture

```
1. Load JSON → Normalize text (NFC for Romanian)
2. Extract placeholders → Get 200 chars context before
3. LLM mapping → Structured prompt with reasoning
4. Fuzzy fallback → Multiple algorithms (partial/token/set)
5. Validate → Check field exists before filling
6. Fill → Preserve exact formatting, handle tables
```

## Example Output

```
[1/6] ✓ Loaded 85 JSON fields
[2/6] ✓ Loaded document (150 paragraphs, 8 tables)
[3/6] ✓ Found 120 placeholders (45 unique)
[4/6] Mapping with LLM...
[MAPPED] '_____' → 'Denumirea / numele ofertantului' (0.92)
  Reason: Context mentions 'numele ofertantului'
✓ LLM mapped 35/45 placeholders
[5/6] Filling placeholders...
  ✓ [LLM] _____... → Denumirea / numele ofertantului (5x)
  ✓ [FUZZY] ........ → Data completarii (2x)
[6/6] ✓ Saved: sample_forms.filled.docx

Fill rate: 88.3%
```

## License

MIT