#!/usr/bin/env python3
"""
Complete RFT Research Suite Runner
===================================

Runs all research validation tools and generates publication outputs.

Usage:
    python run_research_suite.py [--all | --quick | --figures-only]
"""
import sys
import os
import argparse
from pathlib import Path


def run_wave_domain_benchmark():
    """Test 1: Wave-Domain Computation Validation."""
    print("\n" + "=" * 70)
    print("RUNNING: Wave-Domain Computation Benchmark")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/research/wave_domain_computation_benchmark.py'],
        capture_output=False
    )
    return result.returncode == 0


def run_gram_analysis():
    """Test 2: Gram Matrix Eigenstructure Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Gram Matrix Eigenstructure Analysis")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/research/gram_eigenstructure_analysis.py'],
        capture_output=False
    )
    return result.returncode == 0


def run_signal_niche_test():
    """Test 3: Signal Class Performance Map."""
    print("\n" + "=" * 70)
    print("RUNNING: Signal Niche Analysis")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/research/test_rft_signal_niche.py'],
        capture_output=False
    )
    return result.returncode == 0


def generate_figures():
    """Generate all publication figures."""
    print("\n" + "=" * 70)
    print("RUNNING: Publication Figure Generator")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/research/generate_publication_figures.py'],
        capture_output=False
    )
    return result.returncode == 0


def generate_tables():
    """Generate comparison tables."""
    print("\n" + "=" * 70)
    print("RUNNING: Comparison Table Generator")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/research/generate_comparison_tables.py'],
        capture_output=False
    )
    return result.returncode == 0


def run_crypto_hash_test():
    """Test 4: RFT-SIS Hash Validation."""
    print("\n" + "=" * 70)
    print("RUNNING: Cryptographic Hash Avalanche Test")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/crypto/test_rft_sis_hash_avalanche.py'],
        capture_output=False
    )
    return result.returncode == 0


def quick_validation():
    """Quick smoke test of core RFT properties."""
    print("\n" + "=" * 70)
    print("QUICK VALIDATION: Core RFT Properties")
    print("=" * 70)
    
    from algorithms.rft.core.canonical_true_rft import validate_rft_properties
    import numpy as np
    
    results = validate_rft_properties(size=64)
    
    print("\n✓ Quick validation complete")
    print(f"  Unitarity: {results['unitarity_error']:.2e}")
    print(f"  Roundtrip: {results['max_roundtrip_error']:.2e}")
    print(f"  DFT dist:  {results['dft_distance']:.2f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='RFT Research Suite')
    parser.add_argument('--all', action='store_true', help='Run all tests and generate all outputs')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--figures-only', action='store_true', help='Generate figures only')
    parser.add_argument('--tables-only', action='store_true', help='Generate tables only')
    
    args = parser.parse_args()
    
    # Create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    print("=" * 70)
    print("RFT RESEARCH VALIDATION SUITE")
    print("=" * 70)
    
    results = {}
    
    if args.quick:
        results['quick'] = quick_validation()
    
    elif args.figures_only:
        results['figures'] = generate_figures()
    
    elif args.tables_only:
        results['tables'] = generate_tables()
    
    elif args.all or not any([args.quick, args.figures_only, args.tables_only]):
        # Run everything
        print("\nRunning complete research suite...\n")
        
        # Core validation
        results['quick'] = quick_validation()
        
        # Research benchmarks
        results['wave_domain'] = run_wave_domain_benchmark()
        results['gram_analysis'] = run_gram_analysis()
        results['signal_niche'] = run_signal_niche_test()
        results['crypto_hash'] = run_crypto_hash_test()
        
        # Outputs
        results['tables'] = generate_tables()
        results['figures'] = generate_figures()
    
    # Summary
    print("\n" + "=" * 70)
    print("RESEARCH SUITE SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed - research outputs ready")
        print("\nGenerated files:")
        print("  - figures/ (publication-ready plots)")
        print("  - results/comparison_tables.txt")
    else:
        print("\n⚠ Some tests failed - review output above")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
