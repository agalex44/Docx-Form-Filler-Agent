#!/usr/bin/env python
"""
Execute DOCX Form Filler with OpenRouter + QWEN
"""

import os
import sys
from pathlib import Path

from openrouter_agent import DocxFormFillerAgent
from config import (
    INPUT_DOCX_PATH,
    INPUT_DATA_PATH,
    OUTPUT_DOCX_PATH,
    OPENROUTER_API_KEY
)

def validate_setup():
    """Validate environment and files"""
    
    # Check API key
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "REPLACE_WITH_YOUR_KEY":
        print("❌ ERROR: OPENROUTER_API_KEY not set!")
        print("Set it with: export OPENROUTER_API_KEY=your-api-key")
        print("Or add it to .env file")
        sys.exit(1)
    
    print("✓ API key found")
    
    # Check input files
    if not Path(INPUT_DOCX_PATH).exists():
        print(f"❌ ERROR: {INPUT_DOCX_PATH} not found!")
        sys.exit(1)
    
    if not Path(INPUT_DATA_PATH).exists():
        print(f"❌ ERROR: {INPUT_DATA_PATH} not found!")
        sys.exit(1)
    
    print("✓ Input files found")

def main():
    """Main execution"""
    
    print("🚀 Initializing DOCX Form Filler...")
    validate_setup()
    
    # Initialize agent
    agent = DocxFormFillerAgent(use_llm=True)
    
    # Process document
    print("\n📄 Processing document...")
    stats = agent.process(
        docx_path=INPUT_DOCX_PATH,
        data_path=INPUT_DATA_PATH,
        output_path=OUTPUT_DOCX_PATH
    )
    
    print("\n✅ COMPLETE!")
    print(f"Output saved to: {OUTPUT_DOCX_PATH}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
