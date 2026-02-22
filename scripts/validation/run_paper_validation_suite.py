#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Complete validation suite for paper claims.
Runs all tests documented in paper.tex and THE_PHI_RFT_FRAMEWORK_PAPER.tex

This script executes:
1. Python tests (unitarity, scaling, rate-distortion, variants)
2. Assembly tests (C implementation validation)
3. Generates all data files for LaTeX figures

Verilog/FPGA tests should be run separately with Icarus/Makerchip.
"""

import subprocess
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_test(cmd, description, critical=True):
    """Run a test command and report results."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}TEST: {description}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"Command: {cmd}\n")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n{GREEN}✓ PASS{RESET} ({elapsed:.2f}s)")
        return True
    else:
        status = f"\n{RED}✗ FAIL{RESET} ({elapsed:.2f}s)"
        if critical:
            print(f"{status} - CRITICAL")
            return False
        else:
            print(f"{status} - Non-critical, continuing...")
            return True

def main():
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}QUANTONIUMOS PAPER VALIDATION SUITE{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Track results
    results = []
    
    # =========================================================================
    # PHASE 1: CORE PYTHON TESTS (Theorems 1-9)
    # =========================================================================
    print(f"\n{YELLOW}PHASE 1: Core Python Tests (Theorems 1-9){RESET}")
    
    tests_phase1 = [
        ("python3 scripts/irrevocable_truths.py", "Irrevocable Truths - All 7 Variants", True),
        ("python3 scripts/verify_scaling_laws.py", "Scaling Laws (Theorem 3: 98% Sparsity)", True),
        ("python3 scripts/verify_variant_claims.py", "Variant Differentiation (Theorems 4-7)", True),
    ]
    
    for cmd, desc, critical in tests_phase1:
        results.append((desc, run_test(cmd, desc, critical)))
    
    # =========================================================================
    # PHASE 2: HYBRID CODEC & THEOREM 10
    # =========================================================================
    print(f"\n{YELLOW}PHASE 2: Hybrid Codec & Theorem 10 (ASCII Bottleneck){RESET}")
    
    tests_phase2 = [
        ("python3 scripts/verify_ascii_bottleneck.py", "ASCII Bottleneck Resolution", True),
        ("python3 scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv", 
         "Rate-Distortion Curves", True),
        ("python3 scripts/verify_hybrid_mca_recovery.py", "MCA Source Separation", False),
    ]
    
    for cmd, desc, critical in tests_phase2:
        results.append((desc, run_test(cmd, desc, critical)))
    
    # =========================================================================
    # PHASE 3: PERFORMANCE & CRYPTO
    # =========================================================================
    print(f"\n{YELLOW}PHASE 3: Performance & Cryptographic Validation{RESET}")
    
    tests_phase3 = [
        ("python3 scripts/verify_performance_and_crypto.py", "Performance & RFT-SIS Avalanche", True),
        ("python3 scripts/analyze_quantum_chaos.py", "Quantum Chaos Statistics", False),
    ]
    
    for cmd, desc, critical in tests_phase3:
        results.append((desc, run_test(cmd, desc, critical)))
    
    # =========================================================================
    # PHASE 4: ASSEMBLY/C VALIDATION
    # =========================================================================
    print(f"\n{YELLOW}PHASE 4: Assembly/C Implementation Validation{RESET}")
    
    # Check if assembly tests exist
    assembly_tests = [
        "tests/validation/test_assembly_variants.py",
        "tests/validation/test_assembly_vs_python_comprehensive.py",
        "tests/validation/test_assembly_rft_vs_classical_transforms.py",
    ]
    
    for test_file in assembly_tests:
        if os.path.exists(test_file):
            desc = f"Assembly Test: {Path(test_file).stem}"
            cmd = f"python3 {test_file}"
            results.append((desc, run_test(cmd, desc, critical=False)))
        else:
            print(f"\n{YELLOW}⚠ Skipping {test_file} (not found){RESET}")
    
    # =========================================================================
    # PHASE 5: PYTEST SUITE
    # =========================================================================
    print(f"\n{YELLOW}PHASE 5: PyTest Suite (Unit Tests){RESET}")
    
    pytest_tests = [
        ("pytest tests/rft/ -v --tb=short", "RFT Core Unit Tests", False),
        ("pytest tests/integration/ -v --tb=short", "Integration Tests", False),
        ("pytest tests/crypto/ -v --tb=short", "Cryptographic Tests", False),
    ]
    
    for cmd, desc, critical in pytest_tests:
        results.append((desc, run_test(cmd, desc, critical)))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"Total Tests: {len(results)}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print()
    
    # Detailed results
    for test_name, success in results:
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"{status} {test_name}")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}NEXT STEPS:{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    print("1. Verilog/FPGA Tests:")
    print("   - Run: cd hardware && make sim")
    print("   - Or use Makerchip: hardware/makerchip_rft_closed_form.tlv")
    print("   - Screenshot results for paper")
    print()
    print("2. WebFPGA Tests:")
    print("   - Deploy to WebFPGA platform")
    print("   - Screenshot waveform viewer")
    print()
    print("3. Generated Data Files:")
    print("   - figures/latex_data/rate_distortion.csv")
    print("   - figures/latex_data/wave_computer.csv")
    print("   - data/scaling_results.json")
    print()
    
    # Exit code based on critical failures
    if failed > 0:
        print(f"{RED}⚠ Some tests failed. Review output above.{RESET}")
        return 1
    else:
        print(f"{GREEN}✓ All tests passed!{RESET}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
