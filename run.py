#!/usr/bin/env python
"""
Execute Deterministic DOCX Filler v2.0
"""

import sys
from pathlib import Path
from openrouter_agent import DeterministicDocxFiller
from config import (
    INPUT_DOCX_PATH,
    INPUT_DATA_PATH,
    OUTPUT_DOCX_PATH,
    FUZZY_THRESHOLD,
    VERBOSE,
    DETERMINISM_CHECK_ENABLED,
    DETERMINISM_RUNS
)

def validate_files():
    """Validate input files exist"""
    if not Path(INPUT_DOCX_PATH).exists():
        print(f"❌ ERROR: {INPUT_DOCX_PATH} not found")
        sys.exit(1)
    
    if not Path(INPUT_DATA_PATH).exists():
        print(f"❌ ERROR: {INPUT_DATA_PATH} not found")
        sys.exit(1)
    
    print("✓ Input files validated\n")

def test_determinism(agent: DeterministicDocxFiller):
    """Test that processing is deterministic"""
    import hashlib
    
    print("\n" + "=" * 70)
    print("DETERMINISM TEST")
    print("=" * 70)
    
    hashes = []
    for i in range(DETERMINISM_RUNS):
        test_output = f"test_run_{i}.docx"
        agent.process(INPUT_DOCX_PATH, INPUT_DATA_PATH, test_output, verbose=False)
        
        # Hash output file
        with open(test_output, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            hashes.append(file_hash)
        
        Path(test_output).unlink()  # Cleanup
        print(f"Run {i+1}: {file_hash[:16]}...")
    
    if len(set(hashes)) == 1:
        print("\n✓ PASS: All runs produced identical output")
    else:
        print("\n⚠ FAIL: Non-deterministic output detected")
        print(f"Unique hashes: {len(set(hashes))}")
    
    print("=" * 70)

def main():
    """Main execution"""
    
    print("🚀 Deterministic DOCX Filler v2.0\n")
    
    # Validate
    validate_files()
    
    # Initialize agent
    agent = DeterministicDocxFiller(fuzzy_threshold=FUZZY_THRESHOLD)
    
    # Process
    result = agent.process(
        docx_path=INPUT_DOCX_PATH,
        json_path=INPUT_DATA_PATH,
        output_path=OUTPUT_DOCX_PATH,
        verbose=VERBOSE
    )
    
    # Determinism test
    if DETERMINISM_CHECK_ENABLED:
        test_determinism(agent)
    
    # Summary
    success_rate = (result['success'] / result['total_keys'] * 100) if result['total_keys'] > 0 else 0
    print(f"\n✅ SUCCESS RATE: {success_rate:.1f}%")
    print(f"Output: {OUTPUT_DOCX_PATH}")
    
    return 0 if success_rate > 90 else 1

if __name__ == "__main__":
    sys.exit(main())