"""
Configuration for DOCX Form Filler Agent with OpenRouter
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# OpenRouter Configuration
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "REPLACE_WITH_YOUR_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional: For rankings on openrouter.ai (leave empty if not needed)
SITE_URL = os.getenv("SITE_URL", "")
SITE_NAME = os.getenv("SITE_NAME", "")

# Model selection - QWEN 8B
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-embedding-8b")

# Generation configuration
GENERATION_CONFIG = {
    "temperature": 0.0,           # Deterministic
    "top_p": 0.95,
    "max_tokens": 8192,
}

# ============================================================================
# Processing Configuration
# ============================================================================

MAX_RETRIES = 3
CONFIDENCE_THRESHOLD = 0.8
FUZZY_MATCH_THRESHOLD = 80
MAX_PLACEHOLDERS_PER_REQUEST = 100

# ============================================================================
# File Configuration
# ============================================================================

INPUT_DOCX_PATH = "sample_forms.docx"
INPUT_DATA_PATH = "input_date.json"
OUTPUT_DOCX_PATH = "sample_forms.filled.docx"

# Validation settings
RUN_DETERMINISM_CHECK = True
DETERMINISM_RUNS = 3
RUN_UNICODE_CHECK = True

# ============================================================================
# Logging Configuration
# ============================================================================

VERBOSE = True
LOG_FILE = "docx_filler.log"

# ============================================================================
# Feature Flags
# ============================================================================

USE_LLM = True
USE_FUZZY_FALLBACK = True
PRESERVE_FORMATTING = True
HANDLE_ARRAYS = True
