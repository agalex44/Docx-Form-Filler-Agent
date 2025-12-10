"""
Configuration for Deterministic DOCX Filler v2.0
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# File Paths
# ============================================================================

INPUT_DOCX_PATH = "sample_forms.docx"
INPUT_DATA_PATH = "input_date.json"
OUTPUT_DOCX_PATH = "sample_forms.filled.docx"

# ============================================================================
# Matching Configuration
# ============================================================================

# Fuzzy matching threshold (0.0 to 1.0)
# 0.85 = good balance between accuracy and coverage
# 0.90 = stricter, may miss some valid matches
# 0.80 = more lenient, may include false positives
FUZZY_THRESHOLD = 0.85

# ============================================================================
# Processing Options
# ============================================================================

# Show detailed output
VERBOSE = True

# Validation runs for determinism check
DETERMINISM_CHECK_ENABLED = False
DETERMINISM_RUNS = 3

# ============================================================================
# Advanced Options
# ============================================================================

# Handle array/list values in JSON
FORMAT_ARRAYS = True  # Convert ["a", "b"] to "a, b"

# Preserve empty placeholders if no match found
PRESERVE_EMPTY_PLACEHOLDERS = True

# ============================================================================
# Logging
# ============================================================================

LOG_FILE = "docx_filler_v2.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR