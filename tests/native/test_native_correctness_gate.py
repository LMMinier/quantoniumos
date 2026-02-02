#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Native RFT Correctness Gate
===========================

This test MUST pass before any claims about the native C++/ASM pipeline.

Tests:
1. Native forward + inverse roundtrip error < 1e-10
2. Native matches Python implementation (same phase formula)
3. Norm preservation

If any test fails, the native module is NOT equivalent to the canonical RFT.
"""
import sys
import numpy as np

# Gate for CI - if native module not available, skip gracefully
NATIVE_AVAILABLE = False
NATIVE_ERROR = None

try:
    import rftmw_native as rft
    NATIVE_AVAILABLE = True
except ImportError as e:
    NATIVE_ERROR = str(e)
except Exception as e:
    NATIVE_ERROR = str(e)

# Python reference implementation
from algorithms.rft.core.phi_phase_fft_optimized import rft_forward, rft_inverse


def test_native_roundtrip():
    """Native forward + inverse must roundtrip with error < 1e-10."""
    if not NATIVE_AVAILABLE:
        print(f"SKIP: Native module not available ({NATIVE_ERROR})")
        return True  # Don't fail CI if not built
    
    np.random.seed(42)
    
    errors = []
    for n in [64, 128, 256, 512, 1024]:
        x = np.random.randn(n).astype(np.float64)
        x_norm = np.linalg.norm(x)
        
        # Native roundtrip
        X = rft.forward(x)
        x_rec = rft.inverse(X)
        
        # Calculate relative error
        err = np.linalg.norm(x - x_rec) / (x_norm + 1e-15)
        errors.append((n, err))
        
        print(f"  N={n:4d}: roundtrip error = {err:.2e}")
    
    max_err = max(e for _, e in errors)
    
    if max_err > 1e-10:
        print(f"FAIL: Native roundtrip error {max_err:.2e} > 1e-10")
        print("      The native inverse is NOT the inverse of the native forward.")
        return False
    
    print(f"PASS: Native roundtrip error {max_err:.2e} < 1e-10")
    return True


def test_native_matches_python():
    """Native output must match Python reference implementation."""
    if not NATIVE_AVAILABLE:
        print(f"SKIP: Native module not available ({NATIVE_ERROR})")
        return True
    
    np.random.seed(42)
    
    # Test with same input
    x = np.random.randn(256).astype(np.float64)
    
    # Python reference
    X_py = rft_forward(x)
    
    # Native
    X_native = rft.forward(x)
    
    # Compare
    err = np.linalg.norm(X_py - X_native) / (np.linalg.norm(X_py) + 1e-15)
    
    print(f"  Python vs Native forward error: {err:.2e}")
    
    # Note: Different phase formulas will cause mismatch
    # Python: θ = 2πβ·frac(k/φ) + πσk²/n
    # Native: θ = 2π·φ^(k/n)
    # These are DIFFERENT - so we expect mismatch unless we align them
    
    if err > 0.01:  # Allow small differences due to float precision
        print(f"WARNING: Native uses DIFFERENT phase formula than Python")
        print(f"         Python: θ = 2πβ·frac(k/φ) + πσk²/n")
        print(f"         Native: θ = 2π·φ^(k/n)")
        print(f"         These are NOT equivalent transforms.")
        return False
    
    print(f"PASS: Native matches Python reference")
    return True


def test_norm_preservation():
    """Transform should preserve energy (Parseval's theorem)."""
    if not NATIVE_AVAILABLE:
        print(f"SKIP: Native module not available ({NATIVE_ERROR})")
        return True
    
    np.random.seed(42)
    
    for n in [64, 256, 1024]:
        x = np.random.randn(n).astype(np.float64)
        x_norm = np.linalg.norm(x)
        
        X = rft.forward(x)
        X_norm = np.linalg.norm(X)
        
        rel_err = abs(x_norm - X_norm) / (x_norm + 1e-15)
        print(f"  N={n:4d}: ||x||={x_norm:.4f}, ||X||={X_norm:.4f}, rel_err={rel_err:.2e}")
        
        if rel_err > 0.01:
            print(f"FAIL: Norm not preserved (error {rel_err:.2e})")
            return False
    
    print(f"PASS: Norm preserved within 1%")
    return True


def diagnose_roundtrip_issue():
    """Diagnose why native roundtrip fails."""
    if not NATIVE_AVAILABLE:
        return
    
    print("\n=== DIAGNOSING NATIVE ROUNDTRIP ISSUE ===\n")
    
    # Simple test signal
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print(f"Input x: {x}")
    
    # Native forward
    X = rft.forward(x)
    print(f"Native forward X: magnitude max = {np.max(np.abs(X)):.4f}")
    
    # Native inverse
    x_rec = rft.inverse(X)
    print(f"Native inverse x_rec: {x_rec}")
    
    # Error
    err = x - x_rec
    print(f"Error (x - x_rec): {err}")
    print(f"Max abs error: {np.max(np.abs(err)):.4f}")
    
    # Check if it's a scaling issue
    if np.linalg.norm(x_rec) > 0:
        scale = np.linalg.norm(x) / np.linalg.norm(x_rec)
        x_rec_scaled = x_rec * scale
        err_scaled = np.max(np.abs(x - x_rec_scaled))
        print(f"\nIf scaling by {scale:.4f}: max error = {err_scaled:.4f}")
        
        # Check if it's a complex vs real issue
        if np.max(np.abs(np.imag(x_rec))) > 0.01:
            print(f"WARNING: Inverse has imaginary component: max imag = {np.max(np.abs(np.imag(x_rec))):.4f}")


def main():
    print("=" * 60)
    print("NATIVE RFT CORRECTNESS GATE")
    print("=" * 60)
    
    if not NATIVE_AVAILABLE:
        print(f"\nNative module not available: {NATIVE_ERROR}")
        print("This is OK for pure-Python CI runs.")
        print("Native tests will run only when module is built.\n")
        return 0
    
    print(f"\nNative module loaded: AVX2={rft.HAS_AVX2}, FMA={rft.HAS_FMA}\n")
    
    results = []
    
    print("Test 1: Native roundtrip")
    results.append(("roundtrip", test_native_roundtrip()))
    
    print("\nTest 2: Native matches Python")
    results.append(("python_match", test_native_matches_python()))
    
    print("\nTest 3: Norm preservation")
    results.append(("norm", test_norm_preservation()))
    
    # If roundtrip failed, diagnose
    if not results[0][1]:
        diagnose_roundtrip_issue()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ Native module is CORRECT - matches canonical RFT")
        return 0
    else:
        print("\n✗ Native module has CORRECTNESS ISSUES")
        print("  Do NOT claim 'C++/ASM matches Python' until these are fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
