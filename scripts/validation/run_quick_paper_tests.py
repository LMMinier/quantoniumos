#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Quick Paper Validation - Run Key Tests for Paper Claims
This version focuses on tests that actually work in the current codebase.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, desc):
    """Run command and show results."""
    print(f"\n{'='*70}")
    print(f"TEST: {desc}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("QUANTONIUMOS PAPER VALIDATION - QUICK RUN")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Irrevocable Truths (All 7 Variants)
    results['Irrevocable Truths'] = run_cmd(
        "python3 scripts/irrevocable_truths.py",
        "Theorem 1-7: All 7 Unitary Variants"
    )
    
    # Test 2: Scaling Laws (98% Sparsity)
    results['Scaling Laws'] = run_cmd(
        "python3 scripts/verify_scaling_laws.py",
        "Theorem 3: Scaling to 98% Sparsity at N=512"
    )
    
    # Test 3: ASCII Bottleneck
    results['ASCII Bottleneck'] = run_cmd(
        "python3 scripts/verify_ascii_bottleneck.py",
        "Theorem 10: ASCII Bottleneck Resolution"
    )
    
    # Test 4: Variant Claims
    results['Variant Claims'] = run_cmd(
        "python3 scripts/verify_variant_claims.py",
        "Theorems 4-7: Variant Differentiation"
    )
    
    # Test 5: Rate-Distortion
    results['Rate-Distortion'] = run_cmd(
        "python3 scripts/verify_rate_distortion.py",
        "Theorem 10: Rate-Distortion Curves"
    )
    
    # Test 6: Performance & Crypto
    results['Performance'] = run_cmd(
        "python3 scripts/verify_performance_and_crypto.py",
        "Theorem 6: RFT-SIS Cryptographic Avalanche"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL PYTHON TESTS PASSED")
        print("\nNEXT: Run Verilog tests manually:")
        print("  cd hardware && make sim")
        print("  Or use Makerchip: hardware/makerchip_rft_closed_form.tlv")
        return 0
    else:
        print(f"\n⚠ {total - passed} tests failed - see output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
